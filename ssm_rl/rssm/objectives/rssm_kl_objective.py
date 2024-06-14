import torch

from ssm_rl.model_learning.kl_objective import AbstractKLObjective

nn = torch.nn


class RSSMKLObjective(AbstractKLObjective):

    def forward(self,
                smoothed_or_post_states: dict[str, torch.Tensor],
                prior_states: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        if self.distribution == "gaussian":
            return self._kl(lhs_mean=smoothed_or_post_states["mean"], lhs_std=smoothed_or_post_states["std"],
                            rhs_mean=prior_states["mean"], rhs_std=prior_states["std"])
        elif self.distribution == "categorical":
            return self._kl(lhs_probs=smoothed_or_post_states["probs"], rhs_probs=prior_states["probs"])
