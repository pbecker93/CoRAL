import gym
import numpy as np

# Software Engineering yaaay


class OldGymInterfaceWrapper:

    def __init__(self, env: gym.Env):
        self._base_env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._base_env, name)

    def reset(self,
              return_info: bool = False,
              **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self._base_env.reset(**kwargs)
        if return_info:
            return obs, info
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        return obs, reward, terminated or truncated, info
