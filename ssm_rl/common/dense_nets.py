import torch

import ssm_rl.common.activation as act

nn = torch.nn


def build_layers(in_features: int,
                 layer_sizes: list[int],
                 activation: str = "ReLU") -> tuple[nn.ModuleList, int]:

    layers = []
    n_in = in_features
    n_out = n_in
    for layer_size in layer_sizes:
        n_out = layer_size
        layers.append(nn.Linear(in_features=n_in, out_features=n_out))
        layers.append(act.get_activation(activation, shape=(n_out,)))
        n_in = n_out
    return nn.ModuleList(layers), n_out
