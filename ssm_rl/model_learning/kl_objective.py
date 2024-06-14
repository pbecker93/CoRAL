import torch
import math

nn = torch.nn
F = torch.nn.functional
dists = torch.distributions


class _AbstractKLLoss(nn.Module):

    def __init__(self,
                 scale_factor: float,
                 free_nats: float,
                 balanced: bool,
                 alpha: float):
        assert 0. <= alpha <= 1.
        assert free_nats >= 0.
        super(_AbstractKLLoss, self).__init__()
        self.scale_factor = scale_factor
        self._free_nats = free_nats
        self._balanced = balanced
        self._alpha = alpha
        self.is_constraint = False

    def _compute_balanced(self, lhs_kl: torch.Tensor, rhs_kl: torch.Tensor) -> tuple[torch.Tensor, dict]:
        log_dict = {"kl": lhs_kl.mean().detach().cpu().numpy()}
        lhs_kl = lhs_kl.clamp(min=self._free_nats).mean()
        rhs_kl = rhs_kl.clamp(min=self._free_nats).mean()
        return self.scale_factor * (self._alpha * lhs_kl + (1. - self._alpha) * rhs_kl), log_dict

    def _return_kl(self, kl: torch.Tensor) -> tuple[torch.Tensor, dict]:
        log_dict = {"kl": kl.mean().detach().cpu().numpy()}
        return self.scale_factor * kl.clamp(min=self._free_nats).mean(), log_dict


class GaussianKLLoss(_AbstractKLLoss):

    @staticmethod
    def kl(lhs_mean: torch.Tensor,
           lhs_std: torch.Tensor,
           rhs_mean: torch.Tensor,
           rhs_std: torch.Tensor) -> torch.Tensor:
        return dists.kl_divergence(dists.Normal(loc=lhs_mean, scale=lhs_std, validate_args=False),
                                   dists.Normal(loc=rhs_mean, scale=rhs_std, validate_args=False)).sum(dim=-1)

    def forward(self,
                lhs_mean: torch.Tensor,
                lhs_std: torch.Tensor,
                rhs_mean: torch.Tensor,
                rhs_std: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self._balanced:
            return self._compute_balanced(lhs_kl=self.kl(lhs_mean=lhs_mean.detach(), lhs_std=lhs_std.detach(),
                                                         rhs_mean=rhs_mean, rhs_std=rhs_std),
                                          rhs_kl=self.kl(lhs_mean=lhs_mean, lhs_std=lhs_std,
                                                         rhs_mean=rhs_mean.detach(), rhs_std=rhs_std.detach()))
        else:
            return self._return_kl(kl=self.kl(lhs_mean=lhs_mean, lhs_std=lhs_std, rhs_mean=rhs_mean, rhs_std=rhs_std))


class CategoricalKLLoss(_AbstractKLLoss):

    @staticmethod
    def _kl(lhs_probs: torch.Tensor,
            rhs_probs: torch.Tensor) -> torch.Tensor:
        kl_loss = dists.kl_divergence(dists.Categorical(probs=lhs_probs, validate_args=False),
                                      dists.Categorical(probs=rhs_probs, validate_args=False))
        # In the DreamerV2 and V3 code, they wrap the Categorical with a "Independent" wrapper which results in a sum
        # of the KL for the individual categoricals per state
        return kl_loss.sum(dim=-1)

    def forward(self,
                lhs_probs: torch.Tensor,
                rhs_probs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self._balanced:
            return self._compute_balanced(lhs_kl=self._kl(lhs_probs=lhs_probs.detach(), rhs_probs=rhs_probs),
                                          rhs_kl=self._kl(lhs_probs=lhs_probs, rhs_probs=rhs_probs.detach()))
        else:
            return self._return_kl(kl=self._kl(lhs_probs=lhs_probs, rhs_probs=rhs_probs))


class AbstractKLObjective(nn.Module):

    def __init__(self,
                 distribution: str,
                 *args, **kwargs):
        super(AbstractKLObjective, self).__init__()
        self.distribution = distribution
        if distribution == "gaussian":
            self._kl = GaussianKLLoss(*args, **kwargs)
        elif distribution == "categorical":
            self._kl = CategoricalKLLoss(*args, **kwargs)
        else:
            raise NotImplementedError(f"KL objective for {distribution} is not implemented.")

    def forward(self,
                smoothed_or_post_states: dict[str, torch.Tensor],
                prior_states: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def get_scale_factor(self):
        if self.is_constraint:
            return self._kl.get_scale_factor()
        else:
            raise self._kl.scale_factor
