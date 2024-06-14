import torch
from typing import Union

from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.policy_learning.latent_imagination.latent_imagination_policy import LatentImaginationPolicy
from ssm_rl.policy_learning.latent_imagination.latent_imagination_trainer import LatentImaginationPolicyTrainer
from ssm_rl.policy_learning.common.abstract_trainer_factory import AbstractMaxEntropyPolicyTrainerFactory
from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM


class LatentImaginationPolicyTrainerFactory(AbstractMaxEntropyPolicyTrainerFactory):

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = super().get_default_config()

        config.lambda_ = 0.95
        config.discount = 0.99

        config.actor_learning_rate = 8e-5
        config.actor_adam_epsilon = 1e-8
        config.actor_clip_norm = 100.0
        config.actor_weight_decay = 0.0

        config.value_learning_rate = 8e-5
        config.value_adam_epsilon = 1e-8
        config.value_clip_norm = 100.0
        config.value_weight_decay = 0.0

        config.add_subconf("slow_value", ConfigDict())
        config.slow_value.use = False
        config.slow_value.decay = 0.98
        config.slow_value.update_interval = 1
        config.slow_value.reg_factor = 1.0

        config.imagine_horizon = 15
        config.imagine_from_smoothed = False

        config.add_subconf(name="objective",
                           sub_conf=self.objective_factory.get_default_config(finalize_adding=finalize_adding))

        config.eval_interval = 1

        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              policy: LatentImaginationPolicy,
              model: AbstractSSM,
              target_sizes: list[Union[int, tuple[int, int, int]]],
              target_types: list[str],
              config: ConfigDict,
              device: torch.device) -> LatentImaginationPolicyTrainer:

        model_objective = self.objective_factory.build(model=model,
                                                       target_sizes=target_sizes,
                                                       target_types=target_types,
                                                       bypass_mask=policy.bypass_mask,
                                                       config=config.objective,
                                                       device=device)

        return LatentImaginationPolicyTrainer(policy=policy,
                                              model_objective=model_objective,
                                              lambda_=config.lambda_,
                                              discount=config.discount,
                                              imagine_horizon=config.imagine_horizon,
                                              imagine_from_smoothed=config.imagine_from_smoothed,
                                              model_learning_rate=config.model_learning_rate,
                                              model_adam_eps=config.model_adam_epsilon,
                                              model_clip_norm=config.model_clip_norm,
                                              model_weight_decay=config.model_weight_decay,
                                              actor_learning_rate=config.actor_learning_rate,
                                              actor_adam_eps=config.actor_adam_epsilon,
                                              actor_clip_norm=config.actor_clip_norm,
                                              actor_weight_decay=config.actor_weight_decay,
                                              value_learning_rate=config.value_learning_rate,
                                              value_adam_eps=config.value_adam_epsilon,
                                              value_clip_norm=config.value_clip_norm,
                                              value_weight_decay=config.value_weight_decay,
                                              entropy_learning_rate=config.entropy.learning_rate,
                                              entropy_adam_eps=config.entropy.adam_epsilon,
                                              entropy_clip_norm=config.entropy.clip_norm,
                                              entropy_bonus=config.entropy.bonus,
                                              learnable_entropy_bonus=config.entropy.learn_bonus,
                                              entropy_exp_activation=config.entropy.exp_activation,
                                              use_slow_value=config.slow_value.use,
                                              slow_value_decay=config.slow_value.decay,
                                              slow_value_update_interval=config.slow_value.update_interval,
                                              slow_value_reg_factor=config.slow_value.reg_factor)
