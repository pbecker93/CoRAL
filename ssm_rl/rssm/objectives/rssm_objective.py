import torch
from typing import Optional

from ssm_rl.rssm.rssm import RSSM
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective

nn = torch.nn


class RSSMObjective(AbstractModelObjective):

    def compute_states(self,
                       observations: list[torch.Tensor],
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       terminated: torch.Tensor,
                       truncated: torch.Tensor,
                       info: dict,
                       return_prior_states: bool = False,
                       smoothed_states_if_avail: bool = False):
        obs_valid = info.get('obs_valid', None)
        embedded_obs = self.model.encode(observations=observations, obs_valid=obs_valid)
        post_states, prior_states = self.model.transition_model.forward_pass(embedded_obs=embedded_obs,
                                                                             actions=actions,
                                                                             obs_valid=obs_valid)
        if return_prior_states:
            return post_states, prior_states
        else:
            return post_states

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  rewards: torch.Tensor,
                                  terminated: torch.Tensor,
                                  truncated: torch.Tensor,
                                  info: dict,
                                  smoothed_states_if_avail: bool = False) \
            -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        assert len(observations) == len(self._obs_objectives)
        assert not smoothed_states_if_avail
        assert not terminated.any(), NotImplementedError
        obs_valid = info.get('obs_valid', None)
        loss_masks = info.get('loss_mask', None)

        embedded_obs = self.model.encode(observations=observations, obs_valid=obs_valid)
        post_states, prior_states = self.model.transition_model.forward_pass(embedded_obs=embedded_obs,
                                                                             actions=actions,
                                                                             obs_valid=obs_valid)

        obs_losses, reward_loss, kl, inv_dyn_loss, log_dict = \
            self._compute_losses(observations=observations,
                                 embedded_obs=embedded_obs,
                                 actions=actions,
                                 rewards=rewards,
                                 smoothed_or_post_features=self.model.transition_model.get_features(post_states),
                                 prior_features=self.model.transition_model.get_features(prior_states),
                                 smoothed_or_post_states=post_states,
                                 prior_states=prior_states,
                                 obs_valid=obs_valid,
                                 loss_masks=loss_masks)

        neg_elbo = sum(obs_losses) + reward_loss + kl + inv_dyn_loss

        return neg_elbo, log_dict, post_states

    @staticmethod
    def _build_log_dict(obs_log_dicts, reward_dict, kl_dict, inv_dyn_dict):
        log_dict = {}
        for i, d in enumerate(obs_log_dicts):
            log_dict |= {f"obs_loss{i}_{k}": v for k, v in d.items()}
        log_dict |= {f"reward_{k}": v for k, v in reward_dict.items()}
        log_dict |= kl_dict
        log_dict |= {f"inv_dyn_{k}": v for k, v in inv_dyn_dict.items()}
        return log_dict

