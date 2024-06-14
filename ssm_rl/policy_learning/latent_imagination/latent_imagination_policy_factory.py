import torch
from typing import Union

from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.policy_learning.latent_imagination.latent_imagination_policy import ValueFn, LatentImaginationPolicy
from ssm_rl.policy_learning.common.actor import TanhActor
from ssm_rl.policy_learning.common.img_preprocessor import AbstractImgPreprocessor


class LatentImaginationPolicyFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.add_subconf("actor", ConfigDict())
        config.actor.num_layers = 3
        config.actor.layer_size = 300
        config.actor.activation = "ELU"
        config.actor.min_std = 1e-4
        config.actor.init_std = 5.0
        config.actor.mean_scale = 5.0
        config.actor.apply_mean_scale = True

        config.add_subconf("value", ConfigDict())
        config.value.num_layers = 3
        config.value.layer_size = 300
        config.value.activation = "ELU"
        config.value.use_two_hot = False
        config.value.two_hot_lower = -20.0
        config.value.two_hot_upper = 20.0
        config.value.two_hot_num_bins = 255

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(model: AbstractSSM,
              obs_are_images: list[bool],
              obs_sizes: list[Union[int, tuple[int, int, int]]],
              bypass_mask: list[bool],
              img_preprocessor: AbstractImgPreprocessor,
              config: ConfigDict,
              action_space,
              device: torch.device):

        assert not any(bypass_mask), "Bypass mask not supported for Dreamer Policy"

        actor = TanhActor(input_dim=model.feature_size,
                          action_dim=action_space.shape[0],
                          num_layers=config.actor.num_layers,
                          layer_size=config.actor.layer_size,
                          init_std=config.actor.init_std,
                          min_std=config.actor.min_std,
                          min_action=action_space.low,
                          max_action=action_space.high,
                          mean_scale=config.actor.mean_scale,
                          activation=config.actor.activation,
                          apply_dreamer_mean_scale=config.actor.apply_mean_scale).to(device)

        value = ValueFn(in_dim=model.feature_size,
                        num_layers=config.value.num_layers,
                        layer_size=config.value.layer_size,
                        activation=config.value.activation,
                        use_two_hot=config.value.use_two_hot,
                        two_hot_lower=config.value.two_hot_lower,
                        two_hot_upper=config.value.two_hot_upper,
                        two_hot_num_bins=config.value.two_hot_num_bins).to(device)

        return LatentImaginationPolicy(model=model,
                                       actor=actor,
                                       value=value,
                                       action_dim=action_space.shape[0],
                                       obs_are_images=obs_are_images,
                                       img_preprocessor=img_preprocessor,
                                       bypass_mask=bypass_mask,
                                       device=device)
