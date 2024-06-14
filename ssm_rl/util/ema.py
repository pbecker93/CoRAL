from copy import deepcopy

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Model Exponential Moving Average V2 from timm"""

    def __init__(self, orig_model: nn.Module, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.orig_model = orig_model
        self.ema_model = deepcopy(orig_model)
        self.ema_model.eval()
        self.decay = decay

        for parameter in self.ema_model.parameters():
            parameter.requires_grad = False

    def update(self):
        with torch.no_grad():
            for ema_param, orig_param in zip(self.ema_model.parameters(), self.orig_model.parameters()):
                ema_param.mul_(self.decay).add_((1 - self.decay) * orig_param)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
