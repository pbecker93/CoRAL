import torch

from ssm_rl.rssm.transition.abstract_tm import AbstractRSSMTM
import ssm_rl.common.dense_nets as dn
from ssm_rl.util.one_hot_categorical import OneHotCategoricalStraightThrough

nn = torch.nn
F = torch.nn.functional


class CatRSSMActivation(nn.Module):

    def __init__(self,
                 categorical_size: int,
                 num_categorical: int,
                 unimix_factor: float):
        super(CatRSSMActivation, self).__init__()
        self._categorical_size = categorical_size
        self._num_categorical = num_categorical
        self._unimix_factor = unimix_factor

    def forward(self, flat_logits: torch.Tensor) -> torch.Tensor:
        logits = flat_logits.reshape(flat_logits.shape[:-1] + (self._num_categorical, self._categorical_size))
        probs = torch.softmax(logits, dim=-1)
        return self._unimix_factor / self._categorical_size + (1 - self._unimix_factor) * probs


class CatRSSMTM(AbstractRSSMTM):
    """Implementation of the Categorical World Model from DreamerV2"""

    def __init__(self,
                 obs_sizes: list[int],
                 categorical_size: int,
                 num_categorical: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 layer_size: int,
                 build_with_obs_valid: bool,
                 with_obs_pre_layers: bool,
                 activation: str = "ReLU",
                 unimix_factor: float = 0.01):
        super(CatRSSMTM, self).__init__(action_dim=action_dim,
                                        obs_sizes=obs_sizes,
                                        build_with_obs_valid=build_with_obs_valid)
        self._categorical_size = categorical_size
        self._num_categorical = num_categorical
        self._state_dim = categorical_size * num_categorical
        self._rec_state_dim = rec_state_dim
        self._with_obs_pre_layers = with_obs_pre_layers

        assert 0 <= unimix_factor < 1, "unimix_factor must be in [0, 1)"
        self._unimix_factor = unimix_factor

        self._build_predict(action_dim=action_dim,
                            num_layers=num_layers,
                            layer_size=layer_size,
                            activation=activation)

        self._build_update(obs_sizes=obs_sizes,
                           num_layers=num_layers,
                           layer_size=layer_size,
                           activation=activation)

    def _unimix_softmax_activation(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return self._unimix_factor / self._categorical_size + (1 - self._unimix_factor) * probs

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       layer_size: int,
                       activation: str):
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
                                               nn.Linear(in_features=post_last_layer_size,
                                                         out_features=self._state_dim),
                                               CatRSSMActivation(categorical_size=self._categorical_size,
                                                                 num_categorical=self._num_categorical,
                                                                 unimix_factor=self._unimix_factor))

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      layer_size: int,
                      activation: str):
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
        obs_layers, obs_out_size = dn.build_layers(in_features=inpt_size,
                                                   layer_sizes=[layer_size] * num_layers,
                                                   activation=activation)
        self._updt_dist_layer = nn.Sequential(*obs_layers,
                                              nn.Linear(in_features=obs_out_size, out_features=self._state_dim),
                                              CatRSSMActivation(categorical_size=self._categorical_size,
                                                                num_categorical=self._num_categorical,
                                                                unimix_factor=self._unimix_factor))

    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        probs = self._updt_dist_layer(torch.cat(obs + [prior_state["gru_cell_state"]], dim=-1))
        return {"probs": probs,
                "sample": OneHotCategoricalStraightThrough(probs=probs, validate_args=False).rsample(),
                "gru_cell_state": prior_state["gru_cell_state"]}

    def update_with_obs_valid(self,
                              prior_state: dict[str, torch.Tensor],
                              obs: list[torch.Tensor],
                              obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = [o.where(ov, torch.ones_like(o) * self._default_value) for o, ov in zip(obs, obs_valid)]
        obs_valid = [ov.float() for ov in obs_valid]
        probs = self._updt_dist_layer(torch.cat(obs + obs_valid + [prior_state["gru_cell_state"]], dim=-1))
        return {"probs": probs,
                "sample": OneHotCategoricalStraightThrough(probs=probs, validate_args=False).rsample(),
                "gru_cell_state": prior_state["gru_cell_state"]}

    def predict(self, post_state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        trans_input = torch.cat([post_state["sample"].flatten(start_dim=-2, end_dim=-1), action], dim=-1)
        cell_state = self._pred_tm_cell(input=self._pred_pre_layers(trans_input),
                                        hx=post_state["gru_cell_state"])
        probs = self._pred_post_layers(cell_state)
        return {"probs": probs,
                "sample": OneHotCategoricalStraightThrough(probs=probs, validate_args=False).rsample(),
                "gru_cell_state": cell_state}

    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        p = self._pred_tm_cell.weight_ih
        return {"probs": torch.zeros(size=[batch_size, self._num_categorical, self._categorical_size],
                                     dtype=p.dtype, device=p.device),
                "sample": torch.zeros(size=[batch_size, self._num_categorical, self._categorical_size],
                                      dtype=p.dtype, device=p.device),
                "gru_cell_state": torch.zeros(size=[batch_size, self._rec_state_dim], dtype=p.dtype, device=p.device)}

    @property
    def feature_size(self) -> int:
        return self._state_dim + self._rec_state_dim

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_sample = state["sample"].reshape(*(state["sample"].shape[:-2] + (self._state_dim, )))
        return torch.cat([flat_sample, state["gru_cell_state"]], dim=-1)

    @property
    def latent_distribution(self) -> str:
        return "categorical"

