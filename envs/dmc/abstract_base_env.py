from typing import Optional
import numpy as np


class AbstractBaseEnv:

    def render(self, img_size: tuple[int, int], cam: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def get_proprioceptive_position(self, state) -> np.ndarray:
        raise NotImplementedError

    def get_proprioceptive_velocity(self, state) -> np.ndarray:
        raise NotImplementedError

    def get_position(self, state) -> np.ndarray:
        raise NotImplementedError

    def get_state(self, state) -> np.ndarray:
        return np.concatenate([np.atleast_1d(v.ravel()) for v in state.observation.values()], axis=-1)

    @property
    def proprioceptive_pos_size(self):
        raise NotImplementedError

    @property
    def proprioceptive_vel_size(self):
        raise NotImplementedError

    @property
    def position_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        return sum([(1 if len(x.shape) == 0 else x.shape[0]) for x in self.observation_spec().values()])

    @property
    def default_action_repeat(self) -> int:
        raise NotImplementedError

    @property
    def physics(self):
        raise NotImplementedError

    def get_info(self, state):
        raise NotImplementedError

    def observation_spec(self):
        raise NotImplementedError

    def action_dim(self):
        raise NotImplementedError

    def get_brightness(self):
        raise NotImplementedError

