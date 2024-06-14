import torch

import ssm_rl.common.dense_nets as dn

nn = torch.nn


class InverseDynamicsObjective(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 action_dim: int,
                 scale_factor: float,
                 num_layers: int,
                 layer_size: int,
                 activation: str):
        super(InverseDynamicsObjective, self).__init__()
        self._scale_factor = scale_factor
        layers, last_layer_size = dn.build_layers(in_features=2 * feature_dim,
                                                  layer_sizes=[layer_size] * num_layers,
                                                  activation=activation)
        layers.append(nn.Linear(in_features=last_layer_size, out_features=action_dim))
        self._inv_dyn_net = nn.Sequential(*layers)

    def forward(self, state_features: torch.Tensor, target_actions: torch.Tensor) -> tuple[torch.Tensor, dict]:
        state = state_features[:, :-1]
        next_state = state_features[:, 1:]
        predicted_action = self._inv_dyn_net(torch.cat([state, next_state], dim=-1))
        mse = torch.mean((predicted_action - target_actions[:, 1:]) ** 2)
        return self._scale_factor * mse, {"mse": mse.detach().cpu().numpy()}
