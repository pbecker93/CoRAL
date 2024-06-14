import math
import torch
from typing import Union
nn = torch.nn

jit = torch.jit


class DiagGaussActivation(nn.Module):

    def __init__(self,
                 init_var: float,
                 min_var: float) -> None:
        super(DiagGaussActivation, self).__init__()
        self._shift = self._get_shift(init_var=init_var, min_var=min_var)
        self._min_var = min_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self._shift) + self._min_var

    @staticmethod
    def _get_shift(init_var: float,
                   min_var: float) -> float:
        return math.log(math.exp(init_var - min_var) - 1)


class ScaledShiftedSigmoidActivation(nn.Module):

    def __init__(self,
                 init_val: float,
                 min_val: float,
                 max_val: float,
                 steepness: float = 1.0) -> None:
        super(ScaledShiftedSigmoidActivation, self).__init__()
        shift_init = init_val - min_val
        self._scale = max_val - min_val
        self._shift = math.log(shift_init / (self._scale - shift_init))
        self._min_val = min_val
        self._steepness = steepness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale * torch.sigmoid(x * self._steepness + self._shift) + self._min_val


class SaveInstanceNorm2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._norm_layer = nn.InstanceNorm2d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        else:
            return self._norm_layer(x)


class LN_Activation(nn.Module):

    def __init__(self,
                 activation: str,
                 norm_shape: Union[int, tuple[int, ...]],
                 activation_first: bool = True):
        super().__init__()
        if isinstance(norm_shape, tuple) and len(norm_shape) == 3:
            # Naming of normalization is confusing but that's what they do in DreamerV3
            # https://github.com/danijar/dreamerv3/blob/84ecf191d967f787f5cc36298e69974854b0df9c/dreamerv3/nets.py#L593
            c = norm_shape[0]
            self.ln = SaveInstanceNorm2d(num_features=c, affine=True, track_running_stats=False)
        else:
            self.ln = nn.LayerNorm(norm_shape)

        self.activation = getattr(nn, activation)()
        if activation_first:
            self.forward = self._act_ln
        else:
            self.forward = self._ln_act

    def _ln_act(self, x):
        return self.ln(self.activation(x))

    def _act_ln(self, x):
        return self.activation(self.ln(x))


def get_activation(activation: str, shape: Union[int, tuple[int, ...]]) -> nn.Module:
    if "ln_" in activation.lower():
        return LN_Activation(activation=activation[3:], norm_shape=shape, activation_first=False)
    elif "_ln" in activation.lower():
        return LN_Activation(activation=activation[:-3], norm_shape=shape, activation_first=True)
    else:
        act = getattr(nn, activation)()
        assert act is not None, f"Unknown activation: {activation}"
        return act

