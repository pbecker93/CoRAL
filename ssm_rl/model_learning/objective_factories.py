import torch

import ssm_rl.common.activation as act
import ssm_rl.common.dense_nets as dn
from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.model_learning.reconstruction_objective import ReconstructionObjective
from ssm_rl.ssm_interface.encoder_decoder import Decoder
from ssm_rl.common.modules import Reshape
from ssm_rl.model_learning.inverse_dynamics_objective import InverseDynamicsObjective
from ssm_rl.model_learning.ssl_projection import (InfoNCEDirectProjectionObjective,
                                                  InfoNCEDualProjectionObjective)

nn = torch.nn


class ReconstructionObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding=False):
        config = ConfigDict()
        config.scale_factor = 1.0
        config.conv_depth_factor = 32
        config.num_layers = 3
        config.layer_size = 300
        config.conv_activation = "ReLU"
        config.dense_activation = "ELU"
        config.output_std = 1.0
        config.use_prior_features = False
        config.learn_elementwise_std = False
        config.min_output_std = 1e-3
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(input_size,
              target_size,
              target_type: str,
              config: ConfigDict):
        if target_type == "image":
            base_module = ReconstructionObjectiveFactory.build_conv(input_size=input_size,
                                                                    target_size=target_size,
                                                                    conv_depth_factor=config.conv_depth_factor,
                                                                    activation=config.conv_activation)
        else:
            mean_net_layers, last_out_size =\
                dn.build_layers(in_features=input_size,
                                layer_sizes=[config.layer_size] * config.num_layers,
                                activation=config.dense_activation)
            mean_net_layers.append(nn.Linear(last_out_size, target_size))
            base_module = nn.Sequential(*mean_net_layers)

        decoder = Decoder(base_module=base_module)
        decoder.set_symlog(work_with_symlog=False)
        return ReconstructionObjective(decoder=decoder,
                                       target_size=target_size,
                                       scale_factor=config.scale_factor,
                                       output_std=config.output_std,
                                       reconstruct_from_prior=config.use_prior_features,
                                       learn_elementwise_std=config.learn_elementwise_std,
                                       min_std=config.min_output_std)

    @staticmethod
    def build_conv(input_size: int,
                   target_size: tuple[int, int, int],
                   conv_depth_factor: int,
                   activation) -> nn.Module:
        assert target_size[1] == 64 and target_size[2] == 64
        return nn.Sequential(
            # h1
            nn.Linear(in_features=input_size,
                      out_features=4 * 8 * conv_depth_factor),
            Reshape(shape=[-1, 4 * 8 * conv_depth_factor, 1, 1]),
            # h2
            nn.ConvTranspose2d(in_channels=4 * 8 * conv_depth_factor,
                               out_channels=4 * conv_depth_factor,
                               kernel_size=(5, 5),
                               stride=(2, 2)),
            act.get_activation(activation, shape=(4 * conv_depth_factor, 5, 5)),

            # h3
            nn.ConvTranspose2d(in_channels=4 * conv_depth_factor,
                               out_channels=2 * conv_depth_factor,
                               kernel_size=(5, 5),
                               stride=(2, 2)),
            act.get_activation(activation, shape=(2 * conv_depth_factor, 13, 13)),

            # h4
            nn.ConvTranspose2d(in_channels=2 * conv_depth_factor,
                               out_channels=1 * conv_depth_factor,
                               kernel_size=(6, 6),
                               stride=(2, 2)),
            act.get_activation(activation, shape=(conv_depth_factor, 30, 30)),

            # h5
            nn.ConvTranspose2d(in_channels=1 * conv_depth_factor,
                               out_channels=target_size[0],
                               kernel_size=(6, 6),
                               stride=(2, 2)),
        )


class InverseDynamicsObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding=False):
        config = ConfigDict()
        config.scale_factor = 1.0
        config.num_layers = 3
        config.layer_size = 300
        config.activation = "ReLU"
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(feature_dim,
              action_dim,
              config: ConfigDict):
        return InverseDynamicsObjective(feature_dim=feature_dim,
                                        action_dim=action_dim,
                                        scale_factor=config.scale_factor,
                                        layer_size=config.layer_size,
                                        num_layers=config.num_layers,
                                        activation=config.activation)


class AbstractKLObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.scale_factor = 1.0
        config.free_nats = 3.0
        config.balanced = False
        config.alpha = 0.8
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def config_to_kwargs(config: ConfigDict) -> dict:
        return dict(scale_factor=config.scale_factor,
                    free_nats=config.free_nats,
                    balanced=config.balanced,
                    alpha=config.alpha)

    @staticmethod
    def build(*args, **kwargs):
        raise NotImplementedError("Abstract class, not implemented")


class SSLObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.scale_factor = 1.0
        config.type = "info_nce"
        config.projection_dim = 64
        config.state_layer_size = 0
        config.state_num_layers = 0
        config.obs_layer_size = 0
        config.obs_num_layers = 0
        config.activation = "ReLU"
        config.layer_norm = True
        config.layer_norm_affine = True
        config.use_prior_features = True
        config.direct_projection = False

        config.add_subconf("info_nce", ConfigDict())
        config.info_nce.init_inverse_temp = 1.0
        config.info_nce.inverse_temp_lr = 2e-3
        config.info_nce.seq_only_negative = False

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(obs_dim,
              state_dim,
              config: ConfigDict):
        if config.direct_projection:
            common_kwargs = dict(scale_factor=config.scale_factor,
                                 use_prior_features=config.use_prior_features,
                                 obs_dim=obs_dim,
                                 state_dim=state_dim,
                                 state_layer_size=config.state_layer_size,
                                 state_num_layers=config.state_num_layers,
                                 activation=config.activation,
                                 layer_norm=config.layer_norm,
                                 layer_norm_affine=config.layer_norm_affine)
            return InfoNCEDirectProjectionObjective(**common_kwargs,
                                                    init_inverse_temp=config.cpc.init_inverse_temp,
                                                    softmax_over="both",
                                                    seq_only_negatives=config.cpc.seq_only_negative,
                                                    inverse_temp_lr=config.cpc.inverse_temp_lr)

        else:
            common_kwargs = dict(scale_factor=config.scale_factor,
                                 use_prior_features=config.use_prior_features,
                                 obs_dim=obs_dim,
                                 state_dim=state_dim,
                                 projection_dim=config.projection_dim,
                                 state_layer_size=config.state_layer_size,
                                 state_num_layers=config.state_num_layers,
                                 obs_layer_size=config.obs_layer_size,
                                 obs_num_layers=config.obs_num_layers,
                                 activation=config.activation,
                                 layer_norm=config.layer_norm,
                                 layer_norm_affine=config.layer_norm_affine)
            return InfoNCEDualProjectionObjective(**common_kwargs,
                                                  init_inverse_temp=config.info_nce.init_inverse_temp,
                                                  softmax_over="both",
                                                  seq_only_negatives=config.info_nce.seq_only_negative,
                                                  inverse_temp_lr=config.info_nce.inverse_temp_lr)
