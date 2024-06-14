from typing import Optional
import torch

nn = torch.nn


class _TimeDistributed(nn.Module):

    def __init__(self,
                 base_module: nn.Module,
                 default_value: float = 0):
        super(_TimeDistributed, self).__init__()
        self._base_module = base_module
        self._default_value = default_value

    @staticmethod
    def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, seq_length = x.shape[:2]
        bs = batch_size * seq_length
        new_shape = [bs, x.shape[2]] if len(x.shape) == 3 else [bs, x.shape[2], x.shape[3], x.shape[4]]
        return x.reshape(new_shape), batch_size, seq_length

    @staticmethod
    def _unflatten(x: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor:
        if len(x.shape) == 2:
            new_shape = [batch_size, seq_length, x.shape[1]]
        else:
            new_shape = [batch_size, seq_length, x.shape[1], x.shape[2], x.shape[3]]
        return x.reshape(new_shape)

    @staticmethod
    def _get_full(valid: torch.Tensor, mask: torch.Tensor, default_value: float) -> torch.Tensor:
        full = torch.ones(size=mask.size()[:2] + valid.size()[1:], device=valid.device, dtype=valid.dtype)
        full *= default_value
        full[mask] = valid
        return full

    def _td_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.squeeze(dim=-1)
            y_valid = self._base_module(x[mask])
            return self._get_full(y_valid, mask, self._default_value)
        else:
            x_flat, batch_size, seq_length = self._flatten(x)
            y_flat = self._base_module(x_flat)
            return self._unflatten(y_flat, batch_size, seq_length)


class _SymlogEncDec(_TimeDistributed):

    def __init__(self,
                 base_module: nn.Module,
                 default_value: float = 0):
        super().__init__(base_module=base_module, default_value=default_value)
        self._uses_symlog = False
        self._symlog_set = False

    def set_symlog(self, work_with_symlog: bool):
        if not self._symlog_set:
            self._uses_symlog = work_with_symlog
            self._symlog_set = True
        else:
            raise RuntimeError("Symlog can only be set once.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self._symlog_set = True
        return self._forward(*args, **kwargs)

    def _forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    @property
    def uses_symlog(self) -> bool:
        return self._uses_symlog


class Encoder(_SymlogEncDec):

    def __init__(self,
                 base_module: nn.Module,
                 default_value: int = 0):
        super().__init__(base_module=base_module, default_value=default_value)
        self._base_module = base_module
        self._default_value = default_value

    def _forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.symlog(x) if self._uses_symlog else x
        return self._td_forward(x, mask)


class Decoder(_SymlogEncDec):

    def __init__(self,
                 base_module: nn.Module,
                 default_value: int = 0):
        super().__init__(base_module=base_module, default_value=default_value)
        self._base_module = base_module
        self._default_value = default_value

    def _forward(self, x: torch.Tensor, mask: torch.Tensor = None, skip_symexp: bool = False) -> torch.Tensor:
        y = self._td_forward(x, mask)
        return self.symexp(y) if self._uses_symlog and not skip_symexp else y
