import torch
from typing import Union
from ssm_rl.util.config_dict import ConfigDict

from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM

from ssm_rl.policy_learning.common.bypassing import BypassCritic
from ssm_rl.policy_learning.common.actor import TanhActor, TanhBypassActor
from ssm_rl.policy_learning.common.img_preprocessor import AbstractImgPreprocessor

from ssm_rl.policy_learning.actor_critic.actor_critic_policy import ActorCriticPolicy
from ssm_rl.policy_learning.actor_critic.double_q_critic import DoubleQCritic


class _ActorCriticPolicyFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.use_det_features = False
        config.project_for_bypass = False

        config.add_subconf("critic", ConfigDict())
        config.critic.num_layers = 3
        config.critic.layer_size = 300
        config.critic.activation = "ELU"

        if finalize_adding:
            config.finalize_adding()

        return config

    def build(self,
              model: AbstractSSM,
              obs_are_images: list[bool],
              obs_sizes: list[Union[int, tuple[int, int, int]]],
              bypass_mask: list[bool],
              img_preprocessor: AbstractImgPreprocessor,
              config: ConfigDict,
              action_space,
              device: torch.device):

        assert len(obs_are_images) == len(obs_sizes) == len(bypass_mask)

        obs_sizes_for_bypass = [os for os, bypass in zip(obs_sizes, bypass_mask) if bypass]
        if config.project_for_bypass:
            feature_size = model.feature_size if model.feature_size > 0 else config.bypass_project_dim
            factor = sum([1 for bypass in bypass_mask if bypass]) + (0 if all(bypass_mask) else 1)
            representation_dim = factor * feature_size
        else:
            representation_dim = \
                model.feature_size + sum([os for os, bypass in zip(obs_sizes, bypass_mask) if bypass])

        actor = self._build_actor(feature_size=model.feature_size,
                                  representation_dim=representation_dim,
                                  obs_sizes_for_bypass=obs_sizes_for_bypass,
                                  config=config,
                                  action_space=action_space).to(device)

        critic = DoubleQCritic(input_dim=representation_dim,
                               act_dim=action_space.shape[0],
                               num_layers=config.critic.num_layers,
                               layer_size=config.critic.layer_size,
                               activation_fn=config.critic.activation,
                               use_two_hot=False,
                               two_hot_lower=-20,
                               two_hot_upper=20,
                               two_hot_num_bins=255)
        critic = BypassCritic(base=critic,
                              use_projectors=config.project_for_bypass,
                              feature_size=model.feature_size if model.feature_size > 0 else 0,
                              bypass_obs_sizes=obs_sizes_for_bypass).to(device)

        return ActorCriticPolicy(model=model,
                                 actor=actor,
                                 critic=critic,
                                 action_dim=action_space.shape[0],
                                 obs_are_images=obs_are_images,
                                 img_preprocessor=img_preprocessor,
                                 use_deterministic_features=config.use_det_features,
                                 bypass_mask=bypass_mask,
                                 device=device)

    @staticmethod
    def _build_actor(feature_size: int,
                     representation_dim: int,
                     obs_sizes_for_bypass: list[Union[int, tuple[int, int, int]]],
                     config: ConfigDict,
                     action_space):
        raise NotImplementedError


class SACPolicyFactory(_ActorCriticPolicyFactory):

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = _ActorCriticPolicyFactory.get_default_config(finalize_adding=False)

        config.add_subconf("actor", ConfigDict())
        config.actor.num_layers = 3
        config.actor.layer_size = 300
        config.actor.activation = "ELU"
        config.actor.min_std = 1e-4
        config.actor.init_std = 5.0
        config.actor.mean_scale = 5.0
        config.actor.apply_mean_scale = True


        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def _build_actor(feature_size: int,
                     representation_dim: int,
                     obs_sizes_for_bypass: list[Union[int, tuple[int, int, int]]],
                     config: ConfigDict,
                     action_space):

        actor = TanhActor(input_dim=representation_dim,
                          action_dim=action_space.shape[0],
                          num_layers=config.actor.num_layers,
                          layer_size=config.actor.layer_size,
                          init_std=config.actor.init_std,
                          min_std=config.actor.min_std,
                          min_action=action_space.low,
                          max_action=action_space.high,
                          apply_dreamer_mean_scale=config.actor.apply_mean_scale,
                          mean_scale=config.actor.mean_scale,
                          activation=config.actor.activation)
        return TanhBypassActor(base=actor,
                               use_projectors=config.project_for_bypass,
                               feature_size=feature_size,
                               bypass_obs_sizes=obs_sizes_for_bypass)

