import torch
from typing import Optional

from ssm_rl.model_learning.nll_objective import NLLObjective
from ssm_rl.model_learning.abstract_objective import AbstractObservationObjective
from ssm_rl.ssm_interface.encoder_decoder import Decoder
import ssm_rl.common.activation as va

nn = torch.nn


class ReconstructionObjective(AbstractObservationObjective):

    def __init__(self,
                 scale_factor: float,
                 decoder: Decoder,
                 target_size: int,
                 output_std: float,
                 learn_elementwise_std: bool,
                 min_std: float,
                 reconstruct_from_prior: bool,):
        super(ReconstructionObjective, self).__init__()
        self._nll_objective = NLLObjective(scale_factor=scale_factor)
        self._reconstruction_from_prior = reconstruct_from_prior
        self._decoder = decoder
        self._learn_elementwise_std = learn_elementwise_std

        if learn_elementwise_std:
            self._std_activation = va.DiagGaussActivation(init_var=output_std,
                                                          min_var=min_std)
            self._raw_std = nn.Parameter(torch.zeros(1, target_size))
        else:
            self.register_buffer(name="_std",
                                 tensor=output_std * torch.ones(size=(1,)))

    @property
    def is_reconstruction(self) -> bool:
        return True

    def get_std(self):
        if self._learn_elementwise_std:
            return self._std_activation(self._raw_std)
        else:
            return self._std

    def forward(self,
                prior_features: torch.Tensor,
                post_or_smoothed_features: torch.Tensor,
                target: torch.Tensor,
                element_wise_mask: Optional[torch.Tensor] = None,
                sample_wise_mask: Optional[torch.Tensor] = None):
        if self._reconstruction_from_prior:
            input_features = prior_features[:, 1:]
            target = target[:, 1:]
            element_wise_mask = element_wise_mask[:, 1:] if element_wise_mask is not None else None
            sample_wise_mask = sample_wise_mask[:, 1:] if sample_wise_mask is not None else None
        else:
            input_features = post_or_smoothed_features
        if self._decoder.uses_symlog:
            target = self._decoder.symlog(target)
        predicted_mean = self._decoder(input_features, skip_symexp=True)
        return self._nll_objective(target=target,
                                   predicted_mean=predicted_mean,
                                   predicted_std=self.get_std(),
                                   element_wise_mask=element_wise_mask,
                                   sample_wise_mask=sample_wise_mask)

    @property
    def usees_symlog(self):
        return self._decoder.uses_symlog
