import gym
import numpy as np
from envs.common.obs_types import ObsTypes


class AbstractMBRLEnv(gym.Env):

    def __init__(self,
                 obs_type: ObsTypes,
                 image_to_info,
                 full_state_to_info):
        self._image_to_info = image_to_info
        self._full_state_to_info = full_state_to_info
        self._obs_type = obs_type

    # Interface
    def reset(self, eval_mode: bool = False, *args, **kwargs):
        """Resets the environment and returns the initial observation."""
        raise NotImplementedError

    @staticmethod
    def get_obs_are_images(obs_type):
        if obs_type in [ObsTypes.STATE,
                              ObsTypes.PROPRIOCEPTIVE,
                              ObsTypes.PROPRIOCEPTIVE_POSITION]:
            return [False]
        elif obs_type in [ObsTypes.IMAGE, ObsTypes.GRIPPER]:
            return [True]
        elif obs_type in [ObsTypes.IMAGE_PROPRIOCEPTIVE,
                                ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE,
                                ObsTypes.IMAGE_GOAL,
                                ObsTypes.IMAGE_PROPRIOCEPTIVE_GOAL_CAT,
                                ObsTypes.GRIPPER_GOAL,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL_CAT]:
            return [True, False]
        elif obs_type in [ObsTypes.IMAGE_PROPRIOCEPTIVE_GOAL,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL]:
            return [True, False, False]
        elif obs_type in [ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE]:
            return [True, True, False]
        elif obs_type in [ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE_GOAL]:
            return [True, True, False, False]
        else:
            raise AssertionError

    @property
    def obs_are_images(self):
        return self.get_obs_are_images(self._obs_type)

    @property
    def max_seq_length(self):
        """Returns the maximum number of steps taken by the environment."""
        raise NotImplementedError

    @property
    def action_repeat(self):
        """Returns the number of times the environment is stepped for each action."""
        return 1

    # For Distraction Wrapper
    def get_background_mask(self, obs):
        """Returns a mask of the background of the observation which is used for distraction rendering."""
        return None

    # For Distortion Wrapper
    def get_brightness(self) -> float:
        raise NotImplementedError

    # For Movement Primitive Wrapper
    def get_tracking_controller(self, **kwargs):
        """Returns a tracking controller for the environment."""
        raise NotImplementedError

    def get_interface_wrapper_cls(self):
        """Returns the interface wrapper class for the environment."""
        raise NotImplementedError

    @property
    def duration(self):
        """Returns the (maximum) duration of the environment in seconds."""
        raise NotImplementedError

    @property
    def dt(self):
        """Returns the time step of the environment in seconds."""
        raise NotImplementedError

    # Utility
    @staticmethod
    def _get_ld_space(dim):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=float)

    def get_current_obs(self, obs_type: ObsTypes = None):
        raise NotImplementedError

    @property
    def obs_type(self):
        return self._obs_type

    @property
    def img_size(self):
        raise NotImplementedError
