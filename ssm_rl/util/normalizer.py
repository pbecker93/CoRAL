import torch

nn = torch.nn


class AbstractNormalizer(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NoOpNormalizer(AbstractNormalizer):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class EMAStdNormalizer(AbstractNormalizer):

    def __init__(self, decay: float, min_scale: float = 1.0):
        super().__init__()
        self._decay = decay
        self.register_buffer("_ema_std", torch.tensor(1))
        self.register_buffer("_min_scale", torch.tensor(min_scale))
        self._ct = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_on_batch(x)
        return x / torch.maximum(self._ema_std, self._min_scale)

    def _update_on_batch(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.reshape(-1)
            std = torch.std(x)
            decay = min(self._ct / (self._ct + 1), self._decay)
            self._ema_lower = decay * self._ema_std + (1. - decay) * std
            self._ct += 1


class EMAQuantileNormalizer(AbstractNormalizer):

    def __init__(self,
                 decay: float,
                 lower_quantile: float = 0.05,
                 upper_quantile: float = 0.95,
                 min_scale: float = 1.0):
        super().__init__()
        self._decay = decay
        self._lower_quantile = lower_quantile
        self._upper_quantile = upper_quantile
        self.register_buffer("_ema_lower", torch.tensor(0))
        self.register_buffer("_ema_upper", torch.tensor(1))
        self.register_buffer("_min_scale", torch.tensor(min_scale))
        self._ct = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_on_batch(x)
      #  assert self._ema_upper >= self._ema_lower, "EMA upper bound must be greater than lower bound"
        return x / torch.maximum(self._ema_upper - self._ema_lower, self._min_scale)

    def _update_on_batch(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.reshape(-1)
            sorted_x, _ = torch.sort(x, stable=False)
            lower_idx = int(self._lower_quantile * len(x))
            upper_idx = int(self._upper_quantile * len(x))
            lower = sorted_x[lower_idx]
            upper = sorted_x[upper_idx]

            decay = min(self._ct / (self._ct + 1), self._decay)
            self._ema_lower = decay * self._ema_lower + (1. - decay) * lower
            self._ema_upper = decay * self._ema_upper + (1. - decay) * upper
            self._ct += 1
