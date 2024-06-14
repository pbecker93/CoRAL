import torch
from typing import Optional

dists = torch.distributions
nn = torch.nn


class NLLObjective(nn.Module):

    def __init__(self,
                 scale_factor: float):
        super(NLLObjective, self).__init__()
        self._scale_factor = scale_factor

    @staticmethod
    def _mse(target: torch.Tensor,
             predicted: torch.Tensor,
             element_wise_mask: Optional[torch.Tensor] = None,
             sample_wise_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if element_wise_mask is not None:
            assert len(target.shape) == 5
            element_wise_mse = (target - predicted).square() * element_wise_mask.unsqueeze(2)
            num_valid_pixels = torch.sum(element_wise_mask, dim=(-2, -1)) * element_wise_mse.shape[2]
            sample_wise_mse = element_wise_mse.sum(dim=(2, 3, 4)) / num_valid_pixels
        else:
            red_axis = tuple(- (i + 1) for i in range(len(target.shape) - 2))
            sample_wise_mse = (target - predicted).square().mean(red_axis)

        if sample_wise_mask is not None:
            sample_wise_mask = sample_wise_mask.reshape(sample_wise_mse.shape)
            return (sample_wise_mse * sample_wise_mask).sum() / torch.count_nonzero(sample_wise_mask)
        else:
            return sample_wise_mse.mean()

    @staticmethod
    def _gaussian_ll(target: torch.Tensor,
                     predicted_mean: torch.Tensor,
                     predicted_std: torch.Tensor,
                     element_wise_mask: Optional[torch.Tensor] = None,
                     sample_wise_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        element_wise_ll = \
            dists.Normal(loc=predicted_mean, scale=predicted_std, validate_args=False).log_prob(value=target)

        if element_wise_mask is not None:
            assert len(target.shape) == 5
            element_wise_ll = element_wise_ll * element_wise_mask.unsqueeze(2)
            num_valid_pixels = torch.sum(element_wise_mask, dim=(-2, -1)) * element_wise_ll.shape[2]
            num_pixels = target.shape[2] * target.shape[3] * target.shape[4]
            sample_wise_ll = (element_wise_ll.sum(dim=(2, 3, 4)) / num_valid_pixels) * num_pixels
            return sample_wise_ll.mean()
        else:
            sample_wise_ll = element_wise_ll.sum(tuple(- (i + 1) for i in range(len(target.shape) - 2)))

        if sample_wise_mask is not None:
            sample_wise_mask = sample_wise_mask.reshape(sample_wise_ll.shape)
            return (sample_wise_ll * sample_wise_mask).sum() / torch.count_nonzero(sample_wise_mask)
        else:
            return sample_wise_ll.mean()

    def forward(self,
                target: torch.Tensor,
                predicted_mean: torch.Tensor,
                predicted_std: torch.Tensor,
                element_wise_mask: Optional[torch.Tensor] = None,
                sample_wise_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        ll = self._gaussian_ll(target=target,
                               predicted_mean=predicted_mean,
                               predicted_std=predicted_std,
                               element_wise_mask=element_wise_mask,
                               sample_wise_mask=sample_wise_mask)
        with torch.no_grad():
            mse = self._mse(target=target,
                            predicted=predicted_mean,
                            element_wise_mask=element_wise_mask,
                            sample_wise_mask=sample_wise_mask)

        return - self._scale_factor * ll, {'ll': ll.detach().cpu().numpy(), 'mse': mse.cpu().numpy()}

    @property
    def is_ignored(self) -> bool:
        return self._scale_factor == 0
