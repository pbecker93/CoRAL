from typing import Union, Optional
import torch

from ssm_rl.util.ema import EMA
from ssm_rl.util.normalizer import AbstractNormalizer
from ssm_rl.policy_learning.common.abstract_trainer import AbstractMaxEntropyPolicyTrainer
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective
from ssm_rl.policy_learning.actor_critic.actor_critic_policy import ActorCriticPolicy
from ssm_rl.util.freeze_parameters import FreezeParameters

nn = torch.nn


class SACPolicyTrainer(AbstractMaxEntropyPolicyTrainer):

    def __init__(self,
                 policy: ActorCriticPolicy,
                 model_objective: AbstractModelObjective,
                 discount: float,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float,
                 actor_learning_rate: float,
                 actor_adam_eps: float,
                 actor_clip_norm: float,
                 actor_weight_decay: float,
                 critic_learning_rate: float,
                 critic_adam_eps: float,
                 critic_clip_norm: float,
                 critic_weight_decay: float,
                 target_critic_decay: float,
                 target_critic_interval: int,
                 entropy_bonus: float,
                 learnable_entropy_bonus: bool,
                 entropy_learning_rate: float,
                 entropy_adam_eps: float,
                 entropy_clip_norm: float,
                 entropy_exp_activation: bool,
                 target_entropy: Union[float, str] = "auto",
                 detach_states: bool = True):
        super().__init__(model_objective=model_objective,
                         model=policy.model,
                         policy=policy,
                         model_learning_rate=model_learning_rate,
                         model_adam_eps=model_adam_eps,
                         model_clip_norm=model_clip_norm,
                         model_weight_decay=model_weight_decay,
                         entropy_bonus=entropy_bonus,
                         learnable_entropy_bonus=learnable_entropy_bonus,
                         entropy_learning_rate=entropy_learning_rate,
                         entropy_adam_eps=entropy_adam_eps,
                         entropy_clip_norm=entropy_clip_norm,
                         entropy_exp_activation=entropy_exp_activation,
                         target_entropy=target_entropy)

        self._policy = policy
        self._discount = discount

        self._actor_optimizer, self._actor_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.actor.parameters(),
                                               learning_rate=actor_learning_rate,
                                               adam_eps=actor_adam_eps,
                                               clip_norm=actor_clip_norm,
                                               weight_decay=actor_weight_decay)

        critic_parameters = [{'params': self._policy.critic.parameters()}]
        objective_params = self.model_objective.get_parameters_for_optimizer()
        if not detach_states and objective_params is not None:
            critic_parameters.extend(objective_params)

        self._critic_optimizer, self._critic_clip_fn = \
            self._build_optimizer_and_clipping(params=critic_parameters,
                                               learning_rate=critic_learning_rate,
                                               adam_eps=critic_adam_eps,
                                               clip_norm=critic_clip_norm,
                                               weight_decay=critic_weight_decay)

        self._detach_states = detach_states

        self._target_critic = EMA(self._policy.critic, decay=target_critic_decay)
        self._target_critic_update_interval = target_critic_interval
        self._target_critic_update_counter = 0

    def get_optimizer_state_dict(self) -> dict:
        opt_dict = {
            "model": self._model_optimizer.state_dict(),
            "actor": self._actor_optimizer.state_dict(),
            "critic": self._critic_optimizer.state_dict(),
        }
        if self._entropy_bonus.is_trainable:
            opt_dict["entropy"] = self._entropy_optimizer.state_dict()
        return opt_dict

    def load_optimizer_state_dict(self, state_dict: dict):
        self._model_optimizer.load_state_dict(state_dict=state_dict["model"])
        self._actor_optimizer.load_state_dict(state_dict=state_dict["actor"])
        self._critic_optimizer.load_state_dict(state_dict=state_dict["critic"])
        if self._entropy_bonus.is_trainable:
            self._entropy_optimizer.load_state_dict(state_dict=state_dict["entropy"])

    def _get_actor_loss(self, post_states: dict[str, torch.Tensor], obs_for_policy: list[torch.Tensor]):
        current_states = self._policy.get_state_features(post_states)[:, :-1]
        current_obs_for_policy = [obs[:, :-1] for obs in obs_for_policy]
        current_states = current_states.detach()
        policy_actions, policy_action_log_probs = \
            self._policy.actor.get_sampled_action_and_log_prob(ssm_features=current_states,
                                                               bypass_obs=current_obs_for_policy)
        q1, q2 = self._policy.critic(current_states, current_obs_for_policy, policy_actions)
        q = torch.min(q1, q2)
        actor_entropy = - policy_action_log_probs
        actor_loss = - (q.mean() + self._entropy_bonus().detach() * actor_entropy.mean())
        log_dict = {"actor/loss": actor_loss.detach().cpu().numpy(),
                    "actor/entropy": actor_entropy.mean().detach_().cpu().numpy(),
                    "actor/entropy_bonus": self._entropy_bonus().detach_().cpu().numpy()}
        return actor_loss, actor_entropy.detach(), log_dict

    def _update_actor(self, post_states: dict[str, torch.Tensor], obs_for_policy: list[torch.Tensor]):
        freeze_list = [self._policy.critic, self._entropy_bonus]
        if self.model_objective is not None:
            freeze_list += [self.model_objective]
        with FreezeParameters(freeze_list):
            post_states = {k: v.detach() for k, v in post_states.items()}
            actor_loss, actor_entropy, log_dict = self._get_actor_loss(post_states=post_states,
                                                                       obs_for_policy=obs_for_policy)
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_clip_fn()
            self._actor_optimizer.step()
            return actor_entropy, log_dict

    def _update_entropy(self, actor_entropy: torch.Tensor):
        freeze_list = [self._policy.critic, self._policy.critic]
        if self.model_objective is not None:
            freeze_list += [self.model_objective]
        with FreezeParameters(freeze_list):
            if self._entropy_bonus.is_trainable:
                entropy_loss = self._entropy_bonus.compute_loss(actor_entropy=actor_entropy.detach())
                self._entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self._entropy_clip_fn()
                self._entropy_optimizer.step()
                return {"actor/entropy_loss": entropy_loss.detach().cpu().numpy()}
            else:
                return {}

    def _train_on_batch(self, batch, env_step: int = -1, mode: Optional[str] = None) -> dict:

        obs_for_model = [obs for obs, bypass in zip(batch[0], self._policy.bypass_mask) if not bypass]
        obs_for_policy = [obs for obs, bypass in zip(batch[0], self._policy.bypass_mask) if bypass]

        if self._model_optimizer is None:
            assert len(obs_for_model) == 0
            model_obj_log_dict = {}
        else:
            model_loss, model_obj_log_dict = self.model_objective.compute_losses(obs_for_model, *batch[1:])
            loss_dict = {"model/neg_elbo": model_loss.item()}
            model_obj_log_dict = loss_dict | {"model/{}".format(k): v for k, v in model_obj_log_dict.items()}
            self.update_model_parameters(model_loss=model_loss,
                                         retain_graph=not self._detach_states)

        if self._detach_states:
            with torch.no_grad():
                post_states = self.model_objective.compute_states(obs_for_model, *batch[1:],
                                                                  return_prior_states=False,
                                                                  smoothed_states_if_avail=False)
        else:
            post_states = self.model_objective.compute_states(obs_for_model, *batch[1:],
                                                              return_prior_states=False,
                                                              smoothed_states_if_avail=False)

        states = self._policy.get_state_features(post_states)
        if self._detach_states:
            states = states.detach()

        ##########################################################
        # replay buffer for ssm training is o_t, a_t-1, r_t-1
        # for sac we need o_t, a_t, r_t, o_t+1
        ###########################################################
        _, actions, rewards, _, _, _ = batch
        current_states = states[:, :-1]
        current_obs_for_policy = [obs[:, :-1] for obs in obs_for_policy]
        current_actions = actions[:, 1:]
        reward = rewards[:, 1:]
        next_states = states[:, 1:].detach()
        next_obs_for_policy = [obs[:, 1:] for obs in obs_for_policy]

        with torch.no_grad():
            if torch.isnan(next_states).any():
                raise Exception("Next states contains nan")
            next_actions, next_action_log_probs = \
                self._policy.actor.get_sampled_action_and_log_prob(next_states, next_obs_for_policy)
            target_q1, target_q2 = self._target_critic(next_states, next_obs_for_policy, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_v = target_q - self._entropy_bonus().detach() * next_action_log_probs.unsqueeze(-1)
            target_q = reward + self._discount * target_v
        critic_loss = \
            self._policy.critic.compute_loss(current_states, current_obs_for_policy, current_actions, target_q)
        critic_log_dict = {"critic/loss": critic_loss.item()}

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_clip_fn()
        self._critic_optimizer.step()

        with FreezeParameters([self._policy.critic, self._entropy_bonus]):
            actor_entropy, actor_log_dict = self._update_actor(post_states=post_states,
                                                               obs_for_policy=obs_for_policy)
        with FreezeParameters([self._policy.actor, self._policy.critic]):
            entropy_log_dict = self._update_entropy(actor_entropy=actor_entropy)

        self._target_critic_update_counter += 1
        if self._target_critic_update_counter >= self._target_critic_update_interval:
            self._target_critic.update()
            self._target_critic_update_counter = 0
        return model_obj_log_dict | critic_log_dict | actor_log_dict | entropy_log_dict
