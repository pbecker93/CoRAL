import math
import torch
from typing import Optional

import ssm_rl.common.dense_nets as dn
from ssm_rl.model_learning.abstract_objective import AbstractObservationObjective

nn = torch.nn
F = torch.nn.functional


class _AbstractProjectionSSLObjective(AbstractObservationObjective):

    @property
    def is_projection(self) -> bool:
        return True

    @staticmethod
    def _build_projector(input_dim: int,
                         output_dim: int,
                         layer_size: int,
                         num_layers: int,
                         activation: str,
                         layer_norm: bool,
                         layer_norm_affine: bool) -> nn.Module:
        if num_layers > 0:
            layers, last_layer_size = dn.build_layers(in_features=input_dim,
                                                      layer_sizes=[layer_size] * num_layers,
                                                      activation=activation)
        else:
            layers, last_layer_size = [], input_dim
        layers.append(torch.nn.Linear(last_layer_size, output_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(output_dim, elementwise_affine=layer_norm_affine))
        return torch.nn.Sequential(*layers)

    def compute_ssl_loss(self,
                         state_embedding: torch.Tensor,
                         obs_embedding: torch.Tensor,
                         obs_valid: Optional[torch.Tensor] = None):
        raise NotImplementedError


class _AbstractDualProjectionObjective(_AbstractProjectionSSLObjective):

    def __init__(self,
                 scale_factor: float,
                 use_prior_features: bool,
                 obs_dim: int,
                 state_dim: int,
                 projection_dim: int,
                 state_layer_size: int,
                 state_num_layers: int,
                 obs_layer_size: int,
                 obs_num_layers: int,
                 activation: str,
                 layer_norm: bool,
                 layer_norm_affine: bool):
        super().__init__()
        self._scale_factor = scale_factor
        self._use_prior_features = use_prior_features

        self._state_projector = self._build_projector(input_dim=state_dim,
                                                      output_dim=projection_dim,
                                                      layer_size=state_layer_size,
                                                      num_layers=state_num_layers,
                                                      activation=activation,
                                                      layer_norm=layer_norm,
                                                      layer_norm_affine=layer_norm_affine)
        self._obs_projector = self._build_projector(input_dim=obs_dim,
                                                    output_dim=projection_dim,
                                                    layer_size=obs_layer_size,
                                                    num_layers=obs_num_layers,
                                                    activation=activation,
                                                    layer_norm=layer_norm,
                                                    layer_norm_affine=layer_norm_affine)

    def forward(self,
                prior_state_features,
                post_or_smoothed_state_features,
                observation_features,
                obs_valid: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        state_features = prior_state_features if self._use_prior_features else post_or_smoothed_state_features
        if self._use_prior_features:
            state_features = state_features[:, 1:, :]
            observation_features = observation_features[:, 1:, :]
            if obs_valid is not None:
                obs_valid = obs_valid[:, 1:]
        state_embedding = self._state_projector(state_features)
        obs_embedding = self._obs_projector(observation_features)
        ssl_loss, log_dict = self.compute_ssl_loss(state_embedding=state_embedding,
                                                   obs_embedding=obs_embedding,
                                                   obs_valid=obs_valid)
        log_dict["ssl_loss"] = ssl_loss.detach().cpu().numpy()
        return self._scale_factor * ssl_loss, log_dict


class _AbstractDirectProjectionObjective(_AbstractProjectionSSLObjective):

    def __init__(self,
                 scale_factor: float,
                 use_prior_features: bool,
                 obs_dim: int,
                 state_dim: int,
                 state_layer_size: int,
                 state_num_layers: int,
                 activation: str,
                 layer_norm: bool,
                 layer_norm_affine: bool):
        super().__init__()
        self._scale_factor = scale_factor
        self._use_prior_features = use_prior_features

        self._state_projector = self._build_projector(input_dim=state_dim,
                                                      output_dim=obs_dim,
                                                      layer_size=state_layer_size,
                                                      num_layers=state_num_layers,
                                                      activation=activation,
                                                      layer_norm=layer_norm,
                                                      layer_norm_affine=layer_norm_affine)

    def forward(self,
                state_features,
                observation_features,
                obs_valid: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        if self._use_prior_features:
            state_features = state_features[:, 1:, :]
            observation_features = observation_features[:, 1:, :]
            if obs_valid is not None:
                obs_valid = obs_valid[:, 1:]
        state_embedding = self._state_projector(state_features)
        ssl_loss, log_dict = self.compute_ssl_loss(state_embedding=state_embedding,
                                                   obs_embedding=observation_features,
                                                   obs_valid=obs_valid)
        log_dict["ssl_loss"] = ssl_loss.detach().cpu().numpy()
        return self._scale_factor * ssl_loss, log_dict


def _get_info_nce_projection(base_projection_cls):

    class _InfoNCEObjective(base_projection_cls):

        def __init__(self,
                     init_inverse_temp: float,
                     softmax_over: str,
                     inverse_temp_lr: float,
                     seq_only_negatives: bool,
                     **kwargs):
            super(_InfoNCEObjective, self).__init__(**kwargs)
            self._softmax_over = softmax_over
            init_temp = torch.tensor([math.log(math.exp(init_inverse_temp) - 1.0)])
            self._log_inverse_temp = nn.parameter.Parameter(init_temp)
            self._inverse_temp_lr = inverse_temp_lr
            self._seq_only_negatives = seq_only_negatives

        @property
        def inverse_temp(self):
            return F.softplus(self._log_inverse_temp)

        def _info_nce_loss(self, obs_embedding, state_embedding):
            # No additional embedding matrix needed here since both obs and state encoders already have linear output
            # layers, in "logits" entry i,j is the similarity between the i-th observation and the j-th state
            # (B', B')
            logits = self.inverse_temp * torch.matmul(obs_embedding, state_embedding.transpose(dim0=-1, dim1=-2))
            log_probs1 = F.log_softmax(logits, dim=-1)  # marginalize over states
            log_probs2 = F.log_softmax(logits, dim=-2)  # marginalize over observations
            loss1 = -(log_probs1.diagonal().mean())  # infoNCE marginalized over states
            loss2 = -(log_probs2.diagonal().mean())  # infoNCE marginalized over observations
            if self._softmax_over == "states":
                loss = loss1
            elif self._softmax_over == "obs":
                loss = loss2
            else:
                loss = (loss1 + loss2) / 2
            return loss

        def compute_ssl_loss(self,
                             state_embedding: torch.Tensor,
                             obs_embedding: torch.Tensor,
                             obs_valid: Optional[torch.Tensor] = None):
            assert obs_valid is None, "SSL objective does not support observation validity mask - think about this!"
            batch_size, seq_length, *output = state_embedding.shape
            if not self._seq_only_negatives:
                obs_embedding = obs_embedding.reshape(batch_size * seq_length, -1)  # (T * B, D)
                state_embedding = state_embedding.reshape(batch_size * seq_length, -1)  # (T * B, D)
            loss = self._info_nce_loss(obs_embedding=obs_embedding, state_embedding=state_embedding)
            return loss, {"inverse_temp": self.inverse_temp[0].detach().cpu().numpy()}

        def get_parameters_for_optimizer(self):
            param_list = [{"params": self._state_projector.parameters()}]
            if hasattr(self, "_obs_projector"):
                param_list.append({"params": self._obs_projector.parameters()})
            param_list.append({"params": [self._log_inverse_temp], "lr": self._inverse_temp_lr})
            return param_list

    return _InfoNCEObjective


InfoNCEDualProjectionObjective = _get_info_nce_projection(_AbstractDualProjectionObjective)
InfoNCEDirectProjectionObjective = _get_info_nce_projection(_AbstractDirectProjectionObjective)


