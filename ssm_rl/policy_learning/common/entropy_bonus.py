import torch
from typing import Optional
from ssm_rl.common.activation import ScaledShiftedSigmoidActivation

data = torch.utils.data
nn = torch.nn
opt = torch.optim


class EntropyBonus(torch.nn.Module):

    def __init__(self,
                 entropy_bonus: float,
                 learnable: bool,
                 exp_activation: bool,
                 target_entropy: Optional[float] = None):
        super().__init__()
        if learnable:
            assert target_entropy is not None

        self._learnable = learnable

        self._target_entropy = target_entropy

        if self._learnable:
            if exp_activation:
                self._raw_entropy_bonus = torch.nn.Parameter(torch.tensor(entropy_bonus).log())
                self._activation = torch.exp
            else:
                raw_entropy_bonus = torch.tensor(0.0)
                self._raw_entropy_bonus = torch.nn.Parameter(raw_entropy_bonus)
                self._activation = ScaledShiftedSigmoidActivation(init_val=entropy_bonus,
                                                                  min_val=1e-6,
                                                                  max_val=2,
                                                                  steepness=0.5)
        else:
            self.register_buffer("_entropy_bonus", torch.tensor(entropy_bonus))

    def forward(self) -> torch.Tensor:
        return self._activation(self._raw_entropy_bonus) if self._learnable else self._entropy_bonus

    @property
    def is_trainable(self) -> bool:
        return self._learnable

    def compute_loss(self, actor_entropy: torch.Tensor) -> torch.Tensor:
        return self() * (actor_entropy - self._target_entropy).mean()

