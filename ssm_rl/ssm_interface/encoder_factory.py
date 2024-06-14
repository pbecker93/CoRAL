from typing import Tuple
import torch

from ssm_rl.ssm_interface.encoder_decoder import Encoder
from ssm_rl.util.config_dict import ConfigDict
import ssm_rl.common.activation as act
import ssm_rl.common.dense_nets as dn


nn = torch.nn


class EncoderFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True):
        config = ConfigDict()
        config.conv_depth_factor = 32
        config.conv_activation = "ReLU"
        config.dense_activation = "ReLU"
        config.add_dense_layer = False

        config.num_layers = 3
        config.layer_size = 300

        if finalize_adding:
            config.finalize_adding()

        return config

    def build(self,
              input_size,
              input_type,
              config: ConfigDict):
        if input_type == "image":
            base_module, out_size = self.build_conv(input_size=input_size,
                                                    conv_depth_factor=config.conv_depth_factor,
                                                    activation=config.conv_activation,
                                                    add_dense_layer=config.add_dense_layer,
                                                    dense_activation=config.dense_activation,
                                                    dense_out_size=config.layer_size)
        else:
            layers, output_size = \
                dn.build_layers(in_features=input_size,
                                layer_sizes=[config.layer_size] * config.num_layers,
                                activation=config.dense_activation)
            base_module, out_size = nn.Sequential(*layers), output_size
        return Encoder(base_module=base_module), out_size

    @staticmethod
    def build_conv(input_size: tuple[int, int, int],
                   conv_depth_factor,
                   activation,
                   add_dense_layer,
                   dense_activation,
                   dense_out_size) -> Tuple[nn.Module, int]:
        assert len(input_size) == 3
        assert input_size[1] == 64 and input_size[2] == 64
        out_size = 4 * 8 * conv_depth_factor

        """Encoder hidden layers as used in Dreamer, Planet and "World Models" for mujoco_old data_gen"""
        layers = [
            nn.Conv2d(in_channels=input_size[0],
                      out_channels=conv_depth_factor,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            act.get_activation(activation, shape=(conv_depth_factor, 31, 31)),

            nn.Conv2d(in_channels=conv_depth_factor,
                      out_channels=2 * conv_depth_factor,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            act.get_activation(activation, shape=(2 * conv_depth_factor, 14, 14)),

            nn.Conv2d(in_channels=2 * conv_depth_factor,
                      out_channels=4 * conv_depth_factor,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            act.get_activation(activation, shape=(4 * conv_depth_factor, 6, 6)),

            nn.Conv2d(in_channels=4 * conv_depth_factor,
                      out_channels=8 * conv_depth_factor,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            act.get_activation(activation, shape=(8 * conv_depth_factor, 2, 2)),

            nn.Flatten()]

        if add_dense_layer:
            layers.append(nn.Linear(out_size, dense_out_size))
            if dense_activation == "layer_norm":
                layers.append(nn.LayerNorm(dense_out_size))
            else:
                layers.append(act.get_activation(activation, shape=(dense_out_size, )))
        return nn.Sequential(*layers), dense_out_size if add_dense_layer else out_size
