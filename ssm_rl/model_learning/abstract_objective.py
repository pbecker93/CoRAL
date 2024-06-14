import torch
from typing import Optional

from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM

nn = torch.nn


class AbstractModelObjective(nn.Module):

    def __init__(self,
                 model: AbstractSSM,
                 obs_objectives: nn.ModuleList,
                 reward_objective: nn.Module,
                 kl_objective: nn.Module,
                 inverse_dyn_objective: Optional[nn.Module]):
        super().__init__()
        self.model = model
        self._obs_objectives = obs_objectives
        self._reward_objective = reward_objective
        self._kl_objective = kl_objective
        self._inverse_dyn_objective = inverse_dyn_objective

    def compute_losses(self,
                       observations: list[torch.Tensor],
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       terminated: torch.Tensor,
                       truncated: torch.Tensor,
                       info: dict) \
            -> tuple[torch.Tensor, dict]:
        return self.compute_losses_and_states(observations=observations,
                                              actions=actions,
                                              rewards=rewards,
                                              terminated=terminated,
                                              truncated=truncated,
                                              info=info)[:2]

    def compute_states(self,
                       observations: list[torch.Tensor],
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       terminated: torch.Tensor,
                       truncated: torch.Tensor,
                       info: dict,
                       return_prior_states: bool = False,
                       smoothed_states_if_avail: bool = False) \
            -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        raise NotImplementedError

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  rewards: torch.Tensor,
                                  terminated: torch.Tensor,
                                  truncated: torch.Tensor,
                                  info: dict,
                                  smoothed_states_if_avail: bool = False) \
            -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def post_gradient_step_callback(self):
        for obj in self._obs_objectives:
            obj.post_gradient_step_callback()

    def get_parameters_for_optimizer(self):
        if any(hasattr(obj, "get_parameters_for_optimizer") for obj in self._obs_objectives):
            param_list = [{"params": self.model.parameters()},
                          {"params": self._reward_objective.parameters()},
                          {"params": self._kl_objective.parameters()}]

            if self._inverse_dyn_objective is not None:
                param_list.append({"params": self._inverse_dyn_objective.parameters()})

            for obj in self._obs_objectives:
                if hasattr(obj, "get_parameters_for_optimizer"):
                    param_list = param_list + obj.get_parameters_for_optimizer()
                else:
                    param_list.append({"params": obj.parameters()})
            return param_list
        else:
            return self.parameters()

    def _compute_losses(self,
                        observations: list[torch.Tensor],
                        embedded_obs: list[torch.Tensor],
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        smoothed_or_post_features: torch.Tensor,
                        prior_features: torch.Tensor,
                        smoothed_or_post_states: dict[str, torch.Tensor],
                        prior_states: dict[str, torch.Tensor],
                        obs_valid: Optional[list[Optional[torch.Tensor]]],
                        loss_masks: Optional[list[Optional[torch.Tensor]]],

                        ):

        obs_losses, obs_log_dicts = [], []
        for i, (obj, obs) in enumerate(zip(self._obs_objectives, observations)):
            if obj.is_reconstruction:
                loss, log_dict = obj(prior_features=prior_features,
                                     post_or_smoothed_features=smoothed_or_post_features,
                                     target=obs,
                                     sample_wise_mask=obs_valid[i] if obs_valid is not None else None,
                                     element_wise_mask=loss_masks[i] if loss_masks is not None else None)
            elif obj.is_projection:
                loss, log_dict = obj(prior_state_features=prior_features,
                                     post_or_smoothed_state_features=smoothed_or_post_features,
                                     observation_features=embedded_obs[i],
                                     obs_valid=obs_valid[i] if obs_valid is not None else None)
            else:
                raise AssertionError
            obs_losses.append(loss)
            obs_log_dicts.append(log_dict)

        predicted_reward_mean = self.model.reward_decoder(smoothed_or_post_features, skip_symexp=True)
        reward_targets = \
            self.model.reward_decoder.symlog(rewards) if self.model.reward_decoder.uses_symlog else rewards
        reward_loss, reward_log_dict = self._reward_objective(target=reward_targets,
                                                              predicted_mean=predicted_reward_mean,
                                                              predicted_std=torch.ones_like(predicted_reward_mean))

        kl, kl_log_dict = self._kl_objective(smoothed_or_post_states=smoothed_or_post_states,
                                             prior_states=prior_states)
        if self._inverse_dyn_objective is not None:
            inv_dyn_loss, inv_dyn_log_dict = self._inverse_dyn_objective(state_features=smoothed_or_post_features,
                                                                         target_actions=actions)
        else:
            inv_dyn_loss, inv_dyn_log_dict = 0, {}

        log_dict = self._build_log_dict(obs_log_dicts=obs_log_dicts,
                                        reward_dict={} if self._reward_objective.is_ignored else reward_log_dict ,
                                        kl_dict=kl_log_dict,
                                        inv_dyn_dict=inv_dyn_log_dict)
        return obs_losses, reward_loss, kl, inv_dyn_loss, log_dict

    @staticmethod
    def _build_log_dict(obs_log_dicts, reward_dict, kl_dict, inv_dyn_dict):
        log_dict = {}
        for i, d in enumerate(obs_log_dicts):
            log_dict |= {f"obs_loss{i}_{k}": v for k, v in d.items()}
        log_dict |= {f"reward_{k}": v for k, v in reward_dict.items()}
        log_dict |= kl_dict
        log_dict |= {f"inv_dyn_{k}": v for k, v in inv_dyn_dict.items()}
        return log_dict


class AbstractObservationObjective(nn.Module):

    @property
    def is_reconstruction(self) -> bool:
        return False

    @property
    def is_projection(self) -> bool:
        return False

    def post_gradient_step_callback(self):
        pass
