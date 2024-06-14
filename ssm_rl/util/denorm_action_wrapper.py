import torch
import gym


class DenormActionWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 action_mean: torch.Tensor,
                 action_std: torch.Tensor):
        super().__init__(env=env)
        self._action_mean = action_mean
        self._action_std = action_std

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        action = self._action_mean + self._action_std * action
        return self.env.step(action)
