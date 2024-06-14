import torch
from typing import Optional
import ssm_rl.common.dense_nets as dn
from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy
from ssm_rl.policy_learning.common.actor import TanhActor
from ssm_rl.policy_learning.common.img_preprocessor import AbstractImgPreprocessor
from ssm_rl.util.two_hot import TwoHotEncoding

nn = torch.nn
opt = torch.optim

dists = torch.distributions


class ValueFn(nn.Module):

    def __init__(self,
                 in_dim,
                 num_layers: int,
                 layer_size: int,
                 use_two_hot: bool,
                 two_hot_lower: float = -20.0,
                 two_hot_upper: float = 20.0,
                 two_hot_num_bins: int = 255,
                 activation: str = "ReLU"):

        super(ValueFn, self).__init__()

        layers, last_layer_size = dn.build_layers(in_features=in_dim,
                                                  layer_sizes=num_layers * [layer_size],
                                                  activation=activation)
        layers.append(torch.nn.Linear(in_features=last_layer_size,
                                      out_features=two_hot_num_bins if use_two_hot else 1))
        self._v_net = nn.Sequential(*layers)
        self._use_two_hot = use_two_hot
        if use_two_hot:
            self._two_hot = TwoHotEncoding(lower_boundary=two_hot_lower,
                                           upper_boundary=two_hot_upper,
                                           num_bins=two_hot_num_bins)

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self,
                in_features: torch.Tensor,
                return_raw_probs: bool = False) -> torch.Tensor:
        if self._use_two_hot:
            logits = self._v_net(in_features)
            probs = torch.softmax(logits, dim=-1)
            if return_raw_probs:
                return probs
            else:
                return self.symexp(self._two_hot.decode(probs))
        else:
            self._v_net(in_features)

        return self._v_net(in_features)

    def compute_loss(self,
                     in_features: torch.Tensor,
                     targets: torch.Tensor,
                     slow_target_model: Optional[nn.Module],
                     slow_factor: float) -> torch.Tensor:
        if self._use_two_hot:
            logits = self._v_net(in_features)
            probs = torch.softmax(logits, dim=-1)
            targets = self._two_hot.encode(self.symlog(targets))
            target_loss = - (targets * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            if slow_target_model is not None:
                with torch.no_grad():
                    slow_targets = slow_target_model(in_features, return_raw_probs=True)
                slow_loss = -(slow_targets * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            else:
                slow_loss = 0.0
        else:
            prediction = self._v_net(in_features)
            target_loss = 0.5 * (prediction - targets).square().mean()
            if slow_target_model is not None:
                with torch.no_grad():
                    slow_targets = slow_target_model(in_features)
                slow_loss = 0.5 * (prediction - slow_targets).square().mean()
            else:
                slow_loss = 0.0
        return target_loss + slow_factor * slow_loss


class LatentImaginationPolicy(AbstractPolicy, torch.nn.Module):

    def __init__(self,
                 model: AbstractSSM,
                 actor: TanhActor,
                 value: ValueFn,
                 action_dim: int,
                 obs_are_images: list[bool],
                 img_preprocessor: AbstractImgPreprocessor,
                 bypass_mask: list[bool],
                 device: torch.device):
        super(LatentImaginationPolicy, self).__init__(model=model,
                                                      action_dim=action_dim,
                                                      obs_are_images=obs_are_images,
                                                      img_preprocessor=img_preprocessor,
                                                      device=device)
        self.actor = actor
        self.value = value
        self.model = model
        assert not any(bypass_mask), "Bypass mask not supported for AVPolicy"
        self.bypass_mask = bypass_mask

    def _call_internal(self,
                       observation: list[torch.Tensor],
                       prev_action: torch.Tensor,
                       policy_state: dict,
                       sample: bool,
                       obs_valid: Optional[list[torch.Tensor]],
                       step: int = -1  # ignored by AV Policy, here for interface compatibility
                       ) -> tuple[torch.Tensor, dict]:
        post_state = self.model.get_next_posterior(observation=observation,
                                                   action=prev_action,
                                                   post_state=policy_state,
                                                   obs_valid=obs_valid)
        features = self.model.get_features(post_state)
        action = self.actor(features, sample=sample)
        return action, post_state
