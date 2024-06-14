import torch
from collections import OrderedDict
from typing import Union, Optional

from ssm_rl.util.ema import EMA
from ssm_rl.util.normalizer import AbstractNormalizer
from ssm_rl.policy_learning.common.abstract_trainer import AbstractMaxEntropyPolicyTrainer
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective
from ssm_rl.policy_learning.latent_imagination.latent_imagination_policy import LatentImaginationPolicy
from ssm_rl.util.freeze_parameters import FreezeParameters

nn = torch.nn


class LatentImaginationPolicyTrainer(AbstractMaxEntropyPolicyTrainer):

    def __init__(self,
                 policy: LatentImaginationPolicy,
                 model_objective: AbstractModelObjective,
                 discount: float,
                 lambda_: float,
                 imagine_horizon: int,
                 imagine_from_smoothed: bool,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float,
                 actor_learning_rate: float,
                 actor_adam_eps: float,
                 actor_clip_norm: float,
                 actor_weight_decay: float,
                 value_learning_rate: float,
                 value_adam_eps: float,
                 value_clip_norm: float,
                 value_weight_decay: float,
                 use_slow_value: bool,
                 slow_value_decay: float,
                 slow_value_update_interval: int,
                 slow_value_reg_factor: float,
                 entropy_bonus: float,
                 learnable_entropy_bonus: bool,
                 entropy_learning_rate: float,
                 entropy_adam_eps: float,
                 entropy_clip_norm: float,
                 entropy_exp_activation: bool,
                 target_entropy: Union[float, str] = "auto"):
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

        self._imagine_horizon = imagine_horizon
        self._imagine_from_smoothed = imagine_from_smoothed
        self._discount = discount
        self._lambda = lambda_

        self._actor_optimizer, self._actor_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.actor.parameters(),
                                               learning_rate=actor_learning_rate,
                                               adam_eps=actor_adam_eps,
                                               clip_norm=actor_clip_norm,
                                               weight_decay=actor_weight_decay)

        self._value_optimizer, self._value_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.value.parameters(),
                                               learning_rate=value_learning_rate,
                                               adam_eps=value_adam_eps,
                                               clip_norm=value_clip_norm,
                                               weight_decay=value_weight_decay)
        self._use_slow_value = use_slow_value
        if self._use_slow_value:
            self._slow_value = EMA(self._policy.value, decay=slow_value_decay)
            self._slow_value_update_interval = slow_value_update_interval
            self._slow_value_update_counter = 0
            self._slow_value_reg_factor = slow_value_reg_factor
        else:
            self._slow_value_reg_factor = 0.0
            self._slow_value = None

    def get_optimizer_state_dict(self) -> dict:
        opt_dict = {
            "model": self._model_optimizer.state_dict(),
            "actor": self._actor_optimizer.state_dict(),
            "value": self._value_optimizer.state_dict(),
        }
        if self._entropy_bonus.is_trainable:
            opt_dict["entropy"] = self._entropy_optimizer.state_dict()
        return opt_dict

    def load_optimizer_state_dict(self, state_dict: dict):
        self._model_optimizer.load_state_dict(state_dict=state_dict["model"])
        self._actor_optimizer.load_state_dict(state_dict=state_dict["actor"])
        self._value_optimizer.load_state_dict(state_dict=state_dict["value"])
        if self._entropy_bonus.is_trainable:
            self._entropy_optimizer.load_state_dict(state_dict=state_dict["entropy"])

    def _train_on_batch(self, batch, env_step: int = -1, mode: Optional[str] = None) -> OrderedDict:
        # Model Update
        model_loss, model_obj_log_dict, post_states = \
            self.model_objective.compute_losses_and_states(*batch,
                                                           smoothed_states_if_avail=self._imagine_from_smoothed)
        # Actor Update
        with FreezeParameters([self._model, self._policy.value]):
            size = post_states["sample"].shape[0] * post_states["sample"].shape[1]
            initial_states = {k: v.detach().reshape(size, *v.shape[2:]) for k, v in post_states.items()}
            imagined_states, action_log_probs = \
                self._model.rollout_policy(state=initial_states,
                                           policy_fn=self._policy.actor.get_sampled_action_and_log_prob,
                                           num_steps=self._imagine_horizon)
            imagined_features = self._model.get_features(state=imagined_states)
            rewards = self._model.reward_decoder(imagined_features)
            values = self._policy.value(imagined_features)

            lambda_returns = self.compute_generalized_values(rewards=rewards[:, :-1],
                                                             values=values[:, :-1],
                                                             bootstrap=values[:, -1],
                                                             discount=self._discount,
                                                             lambda_=self._lambda)
            actor_entropy = - action_log_probs.mean()
            actor_loss = - (lambda_returns.mean() + self._entropy_bonus().detach() * actor_entropy)

        # Value Update
        with FreezeParameters([self._model, self._policy.actor]):
            with torch.no_grad():
                detached_imagined_features = imagined_features[:, :-1].detach()
                detached_target_values = lambda_returns.detach()
            value_loss = self._policy.value.compute_loss(in_features=detached_imagined_features,
                                                         targets=detached_target_values,
                                                         slow_target_model=self._slow_value,
                                                         slow_factor=self._slow_value_reg_factor)

        with FreezeParameters([self.model_objective, self._policy.actor, self._policy.value]):
            if self._entropy_bonus.is_trainable:
                entropy_loss = self._entropy_bonus.compute_loss(action_log_probs.detach())

        self._model_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()
        self._value_optimizer.zero_grad()

        model_loss.backward()
        actor_loss.backward()
        value_loss.backward()

        self._model_clip_fn()
        self._actor_clip_fn()
        self._value_clip_fn()

        self._model_optimizer.step()
        self._actor_optimizer.step()
        self._value_optimizer.step()

        if self._entropy_bonus.is_trainable:
            self._entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self._entropy_clip_fn()
            self._entropy_optimizer.step()

        if self._use_slow_value:
            self._slow_value_update_counter += 1
            if self._slow_value_update_counter >= self._slow_value_update_interval:
                self._slow_value.update()
                self._slow_value_update_interval = 0

        log_dict = OrderedDict({"model/neg_elbo": model_loss.detach_().cpu().numpy()},
                               **{"model/{}".format(k): v for k, v in model_obj_log_dict.items()})
        log_dict["value/loss"] = value_loss.detach_().cpu().numpy()

        log_dict["actor/loss"] = actor_loss.detach_().cpu().numpy()
        log_dict["actor/entropy"] = actor_entropy.detach_().cpu().numpy()
        log_dict["actor/entropy_bonus"] = self._entropy_bonus().detach_().cpu().numpy()
        if self._entropy_bonus.is_trainable:
            log_dict["actor/entropy_loss"] = entropy_loss.detach().cpu().numpy()
        return log_dict

    @staticmethod
    def compute_generalized_values(rewards: torch.Tensor,
                                   values: torch.Tensor,
                                   bootstrap: torch.Tensor,
                                   discount: float = 0.99,
                                   lambda_: float = 0.95) -> torch.Tensor:
        next_values = torch.cat([values[:, 1:], bootstrap[:, None]], 1)
        deltas = rewards + discount * next_values * (1 - lambda_)
        last = bootstrap
        returns = torch.ones_like(rewards)
        for t in reversed(range(rewards.shape[1])):
            returns[:, t] = last = deltas[:, t] + (discount * lambda_ * last)
        return returns
