import torch
from typing import Optional

from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy
from ssm_rl.policy_learning.common.img_preprocessor import AbstractImgPreprocessor


nn = torch.nn


class _BypassWrapper(nn.Module):

    def __init__(self,
                 base: nn.Module,
                 use_projectors: bool,
                 feature_size: int,
                 bypass_obs_sizes: list[int]):

        super().__init__()
        self._base = base
        self._use_projectors = use_projectors
        if use_projectors:
            self._projectors = \
                nn.ModuleList(nn.Linear(in_features=os, out_features=feature_size) for os in bypass_obs_sizes)

    def _get_features(self,
                      ssm_features: torch.Tensor,
                      bypass_obs: list[torch.Tensor]):
        if self._use_projectors:
            projected_bypass_obs = [proj(obs) for proj, obs in zip(self._projectors, bypass_obs)]
            return torch.cat([ssm_features] + projected_bypass_obs, dim=-1)
        else:
            return torch.cat([ssm_features] + bypass_obs, dim=-1)


class BypassCritic(_BypassWrapper):

    def forward(self,
                ssm_features: torch.Tensor,
                bypass_obs: list[torch.Tensor],
                actions: torch.Tensor):
        return self._base(self._get_features(ssm_features, bypass_obs), actions)

    def compute_loss(self,
                     ssm_features: torch.Tensor,
                     bypass_obs: list[torch.Tensor],
                     actions: torch.Tensor,
                     targets: torch.Tensor):
        return self._base.compute_loss(self._get_features(ssm_features, bypass_obs), actions, targets)


class BypassActor(_BypassWrapper):

    def forward(self,
                ssm_features: torch.Tensor,
                bypass_obs: list[torch.Tensor],
                sample: bool,
                step: int = -1):
        return self._base(in_features=self._get_features(ssm_features, bypass_obs), sample=sample, step=step)


class BypassPolicy(AbstractPolicy, torch.nn.Module):

    def __init__(self,
                 model,
                 actor: BypassActor,
                 action_dim: int,
                 obs_are_images: list[bool],
                 img_preprocessor: AbstractImgPreprocessor,
                 use_deterministic_features: bool,
                 bypass_mask: list[bool],
                 device: torch.device):
        super(BypassPolicy, self).__init__(model=model,
                                           action_dim=action_dim,
                                           obs_are_images=obs_are_images,
                                           img_preprocessor=img_preprocessor,
                                           device=device)
        self.actor = actor
        self.model = model
        self.bypass_mask = bypass_mask
        self.use_deterministic_features = use_deterministic_features

    def _call_internal(self,
                       observation: list[torch.Tensor],
                       prev_action: torch.Tensor,
                       policy_state: dict,
                       sample: bool,
                       obs_valid: Optional[list[torch.Tensor]],
                       step: int = -1) -> tuple[torch.Tensor, dict]:
        obs_for_model = [obs for obs, bypass in zip(observation, self.bypass_mask) if not bypass]
        obs_for_actor = [obs for obs, bypass in zip(observation, self.bypass_mask) if bypass]

        if obs_valid is None:
            obs_valid_for_model = None
        else:
            obs_valid_for_model = [obs for obs, bypass in zip(obs_valid, self.bypass_mask) if not bypass]

        post_state = self.model.get_next_posterior(observation=obs_for_model,
                                                   action=prev_action,
                                                   post_state=policy_state,
                                                   obs_valid=obs_valid_for_model)
        action = self.actor(ssm_features=self.get_state_features(state=post_state),
                            bypass_obs=obs_for_actor,
                            sample=sample,
                            step=step)
        return action, post_state

    def get_state_features(self, state: dict) -> torch.Tensor:
        if self.use_deterministic_features:
            return self.model.get_deterministic_features(state)
        else:
            return self.model.get_features(state)
