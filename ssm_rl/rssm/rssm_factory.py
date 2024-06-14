import torch
from typing import Union

import ssm_rl.common.dense_nets as dn
from ssm_rl.rssm.rssm import RSSM
from ssm_rl.rssm.transition.transition_factory import TransitionFactory
from ssm_rl.ssm_interface.encoder_decoder import Decoder
from ssm_rl.util.config_dict import ConfigDict

nn = torch.nn


class RSSMFactory:

    def __init__(self, encoder_factories):
        self.encoder_factories = encoder_factories

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:

        config = ConfigDict()

        for i, factory in enumerate(self.encoder_factories):
            config.add_subconf(name="encoder{}".format(i),
                               sub_conf=factory.get_default_config(finalize_adding=finalize_adding))

        config.add_subconf(name="transition",
                           sub_conf=TransitionFactory.get_default_config(finalize_adding=finalize_adding))

        config.add_subconf(name="reward_decoder",
                           sub_conf=ConfigDict())
        config.reward_decoder.num_layers = 2
        config.reward_decoder.layer_size = 200
        config.reward_decoder.activation = "ReLU"

        if finalize_adding:
            config.finalize_adding()

        return config

    def _build(self,
               config: ConfigDict,
               input_sizes: list[Union[int, tuple[int, int, int]]],
               input_types: list[str],
               bypass_mask: list[bool],
               action_dim: int,
               with_obs_valid: bool):

        encoders, enc_out_sizes = [], []
        non_bypassed_obs = len(bypass_mask) - sum(bypass_mask)
        assert len(self.encoder_factories) == non_bypassed_obs, \
            "Number of encoders must match number of non-bypassed obs"
        for i, (input_size, input_type, factory) in enumerate(zip(input_sizes, input_types, self.encoder_factories)):
            if not bypass_mask[i]:
                current_config = getattr(config, "encoder{}".format(i))
                enc, enc_out_size = factory.build(input_size=input_size,
                                                  input_type=input_type,
                                                  config=current_config)
                encoders.append(enc)
                enc_out_sizes.append(enc_out_size)

        tm_obs_sizes = enc_out_sizes

        transition_model = TransitionFactory().build(config.transition,
                                                     obs_sizes=tm_obs_sizes,
                                                     action_dim=action_dim,
                                                     with_obs_valid=with_obs_valid)
        reward_layers, reward_last_layer_size = \
            dn.build_layers(in_features=transition_model.feature_size,
                            layer_sizes=[config.reward_decoder.layer_size] * config.reward_decoder.num_layers,
                            activation=config.reward_decoder.activation)
        reward_layers.append(nn.Linear(reward_last_layer_size, 1))
        reward_base_module = nn.Sequential(*reward_layers)
        reward_decoder = Decoder(base_module=reward_base_module)
        return encoders, transition_model, reward_decoder

    def build(self,
              config: ConfigDict,
              input_sizes: list[Union[int, tuple[int, int, int]]],
              input_types: list[str],
              bypass_mask: list[bool],
              action_dim: int,
              with_obs_valid: bool):
        encoders, transition_model, reward_decoder = self._build(config=config,
                                                                 input_sizes=input_sizes,
                                                                 input_types=input_types,
                                                                 bypass_mask=bypass_mask,
                                                                 action_dim=action_dim,
                                                                 with_obs_valid=with_obs_valid)
        return RSSM(encoders=torch.nn.ModuleList(encoders),
                    transition_model=transition_model,
                    reward_decoder=reward_decoder)
