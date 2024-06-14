import torch
import math
import ssm_rl.common.dense_nets as dn
from ssm_rl.policy_learning.common.bypassing import BypassActor

nn = torch.nn
dists = torch.distributions


class _TanhTransform(dists.transforms.Transform):

    domain = dists.constraints.real
    codomain = dists.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size: int = 1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other) -> torch.Tensor:
        return isinstance(other, _TanhTransform)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y.clamp(-0.99999997, 0.99999997))

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - nn.functional.softplus(-2. * x))


class _SquashedNormal(dists.transformed_distribution.TransformedDistribution):

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale

        self.base_dist = dists.Normal(loc=loc, scale=scale, validate_args=False)
        transforms = [_TanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self) -> torch.Tensor:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class AbstractActor(nn.Module):

    def forward(self, in_features: torch.Tensor, sample: bool = True, step: int = -1) -> torch.Tensor:
        raise NotImplementedError


class TanhActor(AbstractActor):

    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 num_layers: int,
                 layer_size: int,
                 init_std: float,
                 min_std: float,
                 mean_scale: float,
                 apply_dreamer_mean_scale: bool,
                 min_action: torch.Tensor,
                 max_action: torch.Tensor,
                 activation: str = "ReLU"):
        super(TanhActor, self).__init__()
        assert (min_action == -1).all() and (max_action == 1).all(), NotImplementedError

        layers, last_layer_size = dn.build_layers(in_features=input_dim,
                                                  layer_sizes=num_layers * [layer_size],
                                                  activation=activation)
        layers.append(torch.nn.Linear(in_features=last_layer_size, out_features=2 * action_dim))
        self._actor_net = nn.Sequential(*layers)

        self._mean_scale = mean_scale
        self._min_std = min_std
        self._raw_init_std = math.log(math.exp(init_std) - 1.0)
        self._apply_dreamer_mean_scale = apply_dreamer_mean_scale
        self.action_dim = action_dim

    def _get_dist(self, in_features: torch.Tensor):
        raw_mean, raw_std = torch.chunk(self._actor_net(in_features), chunks=2, dim=-1)

        # Taken directly from the dreamer (and dreamer v2) implementations
        if self._apply_dreamer_mean_scale:
            mean = self._mean_scale * torch.tanh(raw_mean / self._mean_scale)
        else:
            mean = raw_mean
        std = nn.functional.softplus(raw_std + self._raw_init_std) + self._min_std
        return _SquashedNormal(loc=mean, scale=std)

    def forward(self, in_features: torch.Tensor, sample: bool = True, step: int = -1) -> torch.Tensor:
        dist = self._get_dist(in_features=in_features)
        if sample:
            return dist.rsample()
        else:
            return dist.mean

    def get_sampled_action_and_log_prob(self, in_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._get_dist(in_features=in_features)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, in_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(in_features=in_features)
        return dist.log_prob(actions).sum(dim=-1)


class TanhBypassActor(BypassActor):

    def get_sampled_action_and_log_prob(self,
                                        ssm_features: torch.Tensor,
                                        bypass_obs: list[torch.Tensor]):
        return self._base.get_sampled_action_and_log_prob(self._get_features(ssm_features=ssm_features,
                                                                             bypass_obs=bypass_obs))

    def get_log_prob(self,
                     ssm_features: torch.Tensor,
                     bypass_obs: list[torch.Tensor],
                     actions: torch.Tensor):
        return self._base.get_log_prob(in_features=self._get_features(ssm_features, bypass_obs), actions=actions)
