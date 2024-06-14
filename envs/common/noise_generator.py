import numpy as np
import math
from typing import Optional
import gym


class _WhiteNoise:

    def __init__(self,
                 seed,
                 mu: np.ndarray,
                 sigma: np.ndarray):
        self._random = np.random.RandomState(seed)
        self._mu = mu
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return self._random.normal(size=self._mu.shape) * self._sigma

    def reset(self):
        pass


# from the open ai baselines https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class _OrnsteinUhlenbeckNoise:

    def __init__(self,
                 seed,
                 mu: np.ndarray,
                 sigma: np.ndarray,
                 theta: float = .15,
                 dt: float = 1e-2,
                 x0: Optional[np.ndarray] = None):
        self._random = np.random.RandomState(seed)
        self._theta = theta
        self._mu = mu
        self._sigma = sigma
        self._dt = dt
        self._x0 = x0
        self._x_prev = None
        self.reset()

    def __call__(self) -> np.ndarray:
        x = self._x_prev + self._theta * (self._mu - self._x_prev) * self._dt +\
            self._sigma * math.sqrt(self._dt) * self._random.normal(size=self._mu.shape)
        self._x_prev = x
        return x

    def reset(self):
        self._x_prev = self._x0 if self._x0 is not None else np.zeros_like(self._mu)


def build_noise_generator(seed: int,
                          transition_noise_type: str,
                          transition_noise_std: float,
                          action_spec: gym.spaces.Box):
    if transition_noise_std <= 0.0:
        return None
    elif transition_noise_type == "white":
        return _WhiteNoise(seed=seed,
                           mu=np.zeros(action_spec.shape),
                           sigma=transition_noise_std * np.ones(action_spec.shape))
    elif transition_noise_type == "ou":
        return _OrnsteinUhlenbeckNoise(seed=seed,
                                       mu=np.zeros(action_spec.shape),
                                       sigma=transition_noise_std * np.ones(action_spec.shape))
    else:
        raise AssertionError


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noise_gen = _OrnsteinUhlenbeckNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.05)
    for i in range(10):
        noise_gen.reset()
        noise = [noise_gen() for _ in range(1000)]

        plt.plot(noise)
    plt.grid()
    plt.show()
