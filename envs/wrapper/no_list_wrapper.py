import gym
import numpy as np


class NoListWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        assert len(env.observation_space) == 1
        super(NoListWrapper, self).__init__(env=env)
        self.observation_space = self.env.observation_space[0]

    def reset(self, **kwargs,) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        assert len(obs) == 1
        return obs[0], info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert len(obs) == 1
        return obs[0], reward, terminated, truncated, info
