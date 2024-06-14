import torch
from typing import Optional
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective


class MockObjective(AbstractModelObjective):

    def __init__(self):
        super(MockObjective, self).__init__(
            model=None,
            obs_objectives=[],
            reward_objective=None,
            kl_objective=None,
            inverse_dyn_objective=None)

    def get_parameters_for_optimizer(self):
        return None

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  rewards: torch.Tensor,
                                  terminated: torch.Tensor,
                                  truncated: torch.Tensor,
                                  info: dict,
                                  smoothed_states_if_avail: bool = False):
        return torch.zeros(1, ), {}, {}

    def compute_states(self,
                       observations: list[torch.Tensor],
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       terminated: torch.Tensor,
                       truncated: torch.Tensor,
                       info: dict,
                       return_prior_states: bool = False,
                       smoothed_states_if_avail: bool = False):
        dummy = {"dummy": torch.zeros(actions.shape[0], actions.shape[1], 0, device=actions.device)}
        if return_prior_states:
            return dummy, dummy
        else:
            return dummy
