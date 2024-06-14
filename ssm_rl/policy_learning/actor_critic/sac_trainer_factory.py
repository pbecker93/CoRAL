from typing import Union
import torch

from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.policy_learning.actor_critic.actor_critic_policy import ActorCriticPolicy
from ssm_rl.policy_learning.actor_critic.sac_trainer import SACPolicyTrainer
from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_rl.policy_learning.common.abstract_trainer_factory import AbstractMaxEntropyPolicyTrainerFactory


class SACPolicyTrainerFactory(AbstractMaxEntropyPolicyTrainerFactory):

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = super().get_default_config()

        config.discount = 0.99

        config.actor_learning_rate = 8e-5
        config.actor_adam_epsilon = 1e-8
        config.actor_clip_norm = 100.0
        config.actor_weight_decay = 0.0

        config.critic_learning_rate = 8e-5
        config.critic_adam_epsilon = 1e-8
        config.critic_clip_norm = 100.0
        config.critic_weight_decay = 0.0

        config.target_critic_decay = 0.995
        config.target_critic_interval = 1

        config.critic_grad_to_model = False

        if self.objective_factory is not None:
            config.add_subconf(name="objective",
                               sub_conf=self.objective_factory.get_default_config(finalize_adding=finalize_adding))

        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              policy: ActorCriticPolicy,
              model: AbstractSSM,
              target_sizes: list[Union[int, tuple[int, int, int]]],
              target_types: list[str],
              config: ConfigDict,
              device: torch.device) -> SACPolicyTrainer:
        if self.objective_factory is not None:
            model_objective = self.objective_factory.build(model=model,
                                                           target_sizes=target_sizes,
                                                           target_types=target_types,
                                                           bypass_mask=policy.bypass_mask,
                                                           config=config.objective,
                                                           device=device)
        else:
            model_objective = None

        return SACPolicyTrainer(policy=policy,
                                model_objective=model_objective,
                                discount=config.discount,
                                model_learning_rate=config.model_learning_rate,
                                model_adam_eps=config.model_adam_epsilon,
                                model_clip_norm=config.model_clip_norm,
                                model_weight_decay=config.model_weight_decay,
                                actor_learning_rate=config.actor_learning_rate,
                                actor_adam_eps=config.actor_adam_epsilon,
                                actor_clip_norm=config.actor_clip_norm,
                                actor_weight_decay=config.actor_weight_decay,
                                critic_learning_rate=config.critic_learning_rate,
                                critic_adam_eps=config.critic_adam_epsilon,
                                critic_clip_norm=config.critic_clip_norm,
                                critic_weight_decay=config.critic_weight_decay,
                                entropy_learning_rate=config.entropy.learning_rate,
                                entropy_adam_eps=config.entropy.adam_epsilon,
                                entropy_clip_norm=config.entropy.clip_norm,
                                entropy_bonus=config.entropy.bonus,
                                entropy_exp_activation=config.entropy.exp_activation,
                                learnable_entropy_bonus=config.entropy.learn_bonus,
                                target_critic_decay=config.target_critic_decay,
                                target_critic_interval=config.target_critic_interval,
                                detach_states=not config.critic_grad_to_model,
                                target_entropy=config.entropy.target)
