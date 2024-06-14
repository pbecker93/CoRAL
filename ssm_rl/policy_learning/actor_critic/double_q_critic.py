import torch

import ssm_rl.common.dense_nets as dn
from ssm_rl.util.two_hot import TwoHotEncoding

nn = torch.nn
opt = torch.optim


class DoubleQCritic(nn.Module):
    """Critic network, employs double Q-learning."""

    def __init__(self,
                 input_dim,
                 act_dim,
                 num_layers: int,
                 layer_size: int,
                 use_two_hot: bool,
                 two_hot_lower: float = -20.0,
                 two_hot_upper: float = 20.0,
                 two_hot_num_bins: int = 255,
                 activation_fn: str = "ReLU"):
        super().__init__()

        q1_layers, q1_last_layer_size = dn.build_layers(in_features=input_dim + act_dim,
                                                        layer_sizes=num_layers * [layer_size],
                                                        activation=activation_fn)
        q1_layers.append(torch.nn.Linear(in_features=q1_last_layer_size,
                                         out_features=two_hot_num_bins if use_two_hot else 1))
        self._q1 = nn.Sequential(*q1_layers)
        q2_layers, q2_last_layer_size = dn.build_layers(in_features=input_dim + act_dim,
                                                        layer_sizes=num_layers * [layer_size],
                                                        activation=activation_fn)
        q2_layers.append(torch.nn.Linear(in_features=q2_last_layer_size,
                                         out_features=two_hot_num_bins if use_two_hot else 1))
        self._q2 = nn.Sequential(*q2_layers)

        self._use_two_hot = use_two_hot
        if use_two_hot:
            self._two_hot = TwoHotEncoding(lower_boundary=two_hot_lower,
                                           upper_boundary=two_hot_upper,
                                           num_bins=two_hot_num_bins)

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([states, actions], dim=-1)
        if self._use_two_hot:
            logits1, logits2 = self._q1(inputs), self._q2(inputs)
            probs1, probs2 = torch.softmax(logits1, dim=-1), torch.softmax(logits2, dim=-1)
            return self.symexp(self._two_hot.decode(probs1)), self.symexp(self._two_hot.decode(probs2))
        else:
            return self._q1(inputs), self._q2(inputs)

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([states, actions], dim=-1)
        if self._use_two_hot:
            logits1, logits2 = self._q1(inputs), self._q2(inputs)
            probs1, probs2 = torch.softmax(logits1, dim=-1), torch.softmax(logits2, dim=-1)
            target = self._two_hot.encode(self.symlog(targets))
            return - (target * torch.log(probs1 + 1e-8)).sum(dim=-1).mean() \
                   - (target * torch.log(probs2 + 1e-8)).sum(dim=-1).mean()
        else:
            q1, q2 = self._q1(inputs), self._q2(inputs)
            return (q1 - targets).square().mean() + (q2 - targets).square().mean()
