import math
import torch
from typing import Optional
import ssm_rl.common.modules as mod
import ssm_rl.common.dense_nets as dn
from ssm_rl.rssm.transition.abstract_tm import AbstractRSSMTM

nn = torch.nn
dists = torch.distributions


class RRSSMTM(AbstractRSSMTM):
    """Extended Version of the RSSM used in PlaNet and Dreamer (v1), to recover original model use
       build_with_obs_valid=False and state_part_for_update="d"
    """

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 layer_size: int,
                 min_std: float,
                 build_with_obs_valid: bool,
                 with_obs_pre_layers: bool = False,
                 activation: str = "ReLU"):
        super(RRSSMTM, self).__init__(action_dim=action_dim,
                                      obs_sizes=obs_sizes,
                                      build_with_obs_valid=build_with_obs_valid)
        self._state_dim = state_dim
        self._rec_state_dim = rec_state_dim
        self._min_std = min_std
        self._with_obs_pre_layers = with_obs_pre_layers

        self._build_predict(action_dim=action_dim,
                            num_layers=num_layers,
                            layer_size=layer_size,
                            activation=activation,
                            min_std=min_std)

        self._build_update(obs_sizes=obs_sizes,
                           num_layers=num_layers,
                           layer_size=layer_size,
                           activation=activation,
                           min_std=min_std)

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       layer_size: int,
                       activation: str,
                       min_std: float):
        pre_layers, pre_last_layer_size = dn.build_layers(in_features=self._state_dim + action_dim,
                                                          layer_sizes=[layer_size] * num_layers,
                                                          activation=activation)
        self._pred_pre_layers = nn.Sequential(*pre_layers)
        self._pred_tm_cell = nn.GRUCell(input_size=pre_last_layer_size,
                                        hidden_size=self._rec_state_dim)

        post_layers, post_last_layer_size = dn.build_layers(in_features=self._rec_state_dim,
                                                            layer_sizes=[layer_size] * num_layers,
                                                            activation=activation)
        self._pred_post_layers = nn.Sequential(*post_layers,
                                               mod.SimpleGaussianParameterLayer(in_features=post_last_layer_size,
                                                                                distribution_dim=self._state_dim,
                                                                                min_std_or_var=min_std))

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      layer_size: int,
                      activation: str,
                      min_std: float):

        if self._with_obs_pre_layers:
            assert len(obs_sizes) == 1, "Only one observation supported for now"
            assert not self._built_with_obs_valid, "Not supported yet"
            obs_layers, obs_last_layer_size = dn.build_layers(in_features=obs_sizes[0],
                                                              layer_sizes=[layer_size] * num_layers,
                                                              activation=activation)
            self._obs_pre_net = nn.Sequential(*obs_layers,
                                              nn.LayerNorm(obs_last_layer_size))
            inpt_size = obs_last_layer_size + self._rec_state_dim
        else:
            inpt_size = sum(obs_sizes) + (len(obs_sizes) if self._built_with_obs_valid else 0) + self._rec_state_dim
            self._obs_pre_net = None
        layers, out_size = dn.build_layers(in_features=inpt_size,
                                           layer_sizes=[layer_size] * num_layers,
                                           activation=activation)
        self._updt_dist_layer = nn.Sequential(*layers,
                                              mod.SimpleGaussianParameterLayer(in_features=out_size,
                                                                               distribution_dim=self._state_dim,
                                                                               min_std_or_var=min_std))

    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._with_obs_pre_layers:
            obs = [self._obs_pre_net(obs[0])]
        mean, std = self._updt_dist_layer(torch.cat(obs + [prior_state["gru_cell_state"]], dim=-1))
        return {"mean": mean,
                "std": std,
                "sample": dists.Normal(loc=mean, scale=std, validate_args=False).rsample(),
                "gru_cell_state": prior_state["gru_cell_state"]}

    def update_with_obs_valid(self,
                              prior_state: dict[str, torch.Tensor],
                              obs: list[torch.Tensor],
                              obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = [o.where(ov, torch.ones_like(o) * self._default_value) for o, ov in zip(obs, obs_valid)]
        obs_valid = [ov.float() for ov in obs_valid]
        mean, std = self._updt_dist_layer(torch.cat(obs + obs_valid + [prior_state["gru_cell_state"]], dim=-1))
        return {"mean": mean,
                "std": std,
                "sample": dists.Normal(loc=mean, scale=std, validate_args=False).rsample(),
                "gru_cell_state": prior_state["gru_cell_state"]}

    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        trans_input = post_state["sample"] if action is None else torch.cat([post_state["sample"], action], dim=-1)
        cell_state = self._pred_tm_cell(self._pred_pre_layers(trans_input),
                                        post_state["gru_cell_state"])
        mean, std = self._pred_post_layers(cell_state)
        return {"mean": mean,
                "std": std,
                "sample": dists.Normal(loc=mean, scale=std, validate_args=False).rsample(),
                "gru_cell_state": cell_state}

    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        p = self._pred_tm_cell.weight_ih
        return {"mean": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                # this should never be used in the rssm implementation
                "std": - torch.ones(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "sample": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "gru_cell_state": torch.zeros(size=[batch_size, self._rec_state_dim],
                                              device=p.device,
                                              dtype=p.dtype)}

    @property
    def feature_size(self):
        return self._state_dim + self._rec_state_dim

    @property
    def hist_size(self):
        return self._rec_state_dim

    @property
    def latent_size(self):
        return self._state_dim

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["sample"], state["gru_cell_state"]], dim=-1)

    def get_deterministic_features(self, state: dict[str, torch.Tensor]):
        return torch.cat([state["mean"], state["gru_cell_state"]], dim=-1)

    @property
    def latent_distribution(self) -> str:
        return "gaussian"
