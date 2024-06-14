from typing import Union, Optional
import gym

import numpy as np
from envs.common.noise_generator import build_noise_generator
from envs.common.obs_types import ObsTypes
from envs.common.abstract_mbrl_env import AbstractMBRLEnv
from envs.dmc.abstract_base_env import AbstractBaseEnv


class DMCMBRLEnv(AbstractMBRLEnv):

    SUPPORTED_OBS_TYPES = [ObsTypes.IMAGE,
                           ObsTypes.STATE,
                           ObsTypes.POSITION,
                           ObsTypes.PROPRIOCEPTIVE,
                           ObsTypes.PROPRIOCEPTIVE_POSITION,
                           ObsTypes.IMAGE_STATE,
                           ObsTypes.IMAGE_POSITION,
                           ObsTypes.IMAGE_PROPRIOCEPTIVE,
                           ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION]

    def __init__(self,
                 base_env: AbstractBaseEnv,
                 seed: int,
                 obs_type,
                 action_repeat: int = -1,
                 transition_noise_std: float = 0.0,
                 transition_noise_type: str = "white",
                 img_size: tuple[int, int] = (64, 64),
                 image_to_info: bool = False,
                 full_state_to_info: bool = False):

        super(DMCMBRLEnv, self).__init__(obs_type=obs_type,
                                         image_to_info=image_to_info,
                                         full_state_to_info=full_state_to_info)
        assert obs_type in self.SUPPORTED_OBS_TYPES, f"Unsupported observation type: {obs_type}"

        self._seed = seed
        self._base_env = base_env
        self._action_repeat = self._base_env.default_action_repeat if action_repeat < 0 else action_repeat
        self._transition_noise_std = transition_noise_std
        self._img_size = img_size
        self._current_step = 0
        self._fig = None
        self.action_space = self.get_action_space()
        self.action_space.seed(seed=seed)
        [os.seed(seed) for os in self.observation_space]

        self._transition_noise_generator = build_noise_generator(seed=seed,
                                                                 transition_noise_type=transition_noise_type,
                                                                 transition_noise_std=transition_noise_std,
                                                                 action_spec=self.action_space)
        self._current_state = None
        
    def _get_obs(self, state, obs_type=None) -> list[np.ndarray]:
        if obs_type is None:
            obs_type = self._obs_type
        if obs_type == ObsTypes.IMAGE:
            return [self._base_env.render(self._img_size)]
        elif obs_type == ObsTypes.STATE:
            return [self._base_env.get_state(state=state)]
        elif obs_type == ObsTypes.POSITION:
            return [self._base_env.get_position(state=state)]
        elif obs_type == ObsTypes.PROPRIOCEPTIVE:
            return [np.concatenate([self._base_env.get_proprioceptive_position(state=state),
                                    self._base_env.get_proprioceptive_velocity(state=state)], axis=-1)]
        elif obs_type == ObsTypes.PROPRIOCEPTIVE_POSITION:
            return [self._base_env.get_proprioceptive_position(state=state)]
        elif obs_type == ObsTypes.IMAGE_STATE:
            return [self._base_env.render(self._img_size), self._base_env.get_state(state=state)]
        elif obs_type == ObsTypes.IMAGE_POSITION:
            return [self._base_env.render(self._img_size), self._base_env.get_position(state=state)]
        elif obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE:
            return [self._base_env.render(self._img_size),
                    np.concatenate([self._base_env.get_proprioceptive_position(state=state),
                                    self._base_env.get_proprioceptive_velocity(state=state)], axis=-1)]
        elif obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION:
            return [self._base_env.render(self._img_size), self._base_env.get_proprioceptive_position(state=state)]
        else:
            raise AssertionError

    def _get_info(self, state, obs):
        info = self._base_env.get_info(state)
        if self._image_to_info:
            if any(obs_type in self._obs_type for obs_type in [ObsTypes.IMAGE_STATE,
                                                               ObsTypes.IMAGE_POSITION,
                                                               ObsTypes.IMAGE_PROPRIOCEPTIVE,
                                                               ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION]):
                info |= {"image": obs[0].copy()}
            else:
                info |= {"image": self._base_env.render(self._img_size)}
        if self._full_state_to_info:
            info |= {"state": self._base_env.get_state(state=state)}
        return info

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = True,
              options: Optional[dict] = None,
              *args, **kwargs) -> Union[gym.core.ObsType, tuple[gym.core.ObsType, dict]]:
        assert return_info, "Always call reset with return_info=True for MBRL environments"
        if self._transition_noise_generator is not None:
            self._transition_noise_generator.reset()
        state = self._base_env.reset()
        self._current_state = state
        self._current_step = 0
        obs = self._get_obs(state)
        return obs, self._get_info(state=state, obs=obs)

    def step(self, action: np.ndarray) -> tuple[gym.core.ObsType, float, bool, bool, dict]:
        if self._transition_noise_generator is not None:
            action = action + self._transition_noise_generator()
        action = np.clip(a=action, a_min=self.action_space.low, a_max=self.action_space.high)
        reward = 0
        self._current_step += 1
        state = None
        for k in range(self._action_repeat):
            state = self._base_env.step(action)
            self._current_state = state
            reward += state.reward
            if state.last():
                terminated = state.discount == 0
                truncated = state.discount > 0
            else:
                terminated, truncated = False, False

            if terminated or truncated:
                break
        obs = self._get_obs(state)
        return obs, reward, terminated, truncated, self._get_info(state=state, obs=obs)


    @property
    def action_dim(self) -> int:
        return self._base_env.action_spec().shape[0]

    def _get_img_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(self._img_size[0], self._img_size[1], 3), dtype=np.uint8)

    @property
    def observation_space(self):
        if self._obs_type == ObsTypes.IMAGE:
            return gym.spaces.Tuple([self._get_img_space()])
        elif self._obs_type == ObsTypes.STATE:
            return gym.spaces.Tuple([self._get_ld_space(dim=self._base_env.state_size)])
        elif self._obs_type == ObsTypes.POSITION:
            return gym.spaces.Tuple([self._get_ld_space(dim=self._base_env.position_size)])
        elif self._obs_type == ObsTypes.PROPRIOCEPTIVE:
            dim = self._base_env.proprioceptive_pos_size + self._base_env.proprioceptive_vel_size
            return gym.spaces.Tuple([self._get_ld_space(dim=dim)])
        elif self._obs_type == ObsTypes.PROPRIOCEPTIVE_POSITION:
            dim = self._base_env.proprioceptive_pos_size
            return gym.spaces.Tuple([self._get_ld_space(dim=dim)])
        elif self._obs_type == ObsTypes.IMAGE_STATE:
            return gym.spaces.Tuple([self._get_img_space(), self._get_ld_space(dim=self._base_env.state_size)])
        elif self._obs_type == ObsTypes.IMAGE_POSITION:
            ld_space = self._get_ld_space(self._base_env.position_size)
            return gym.spaces.Tuple([self._get_img_space(), ld_space])
        elif self._obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE:
            dim = self._base_env.proprioceptive_pos_size + self._base_env.proprioceptive_vel_size
            ld_space = self._get_ld_space(dim=dim)
            return gym.spaces.Tuple([self._get_img_space(), ld_space])
        elif self._obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION:
            ld_space = self._get_ld_space(self._base_env.proprioceptive_pos_size)
            return [self._get_img_space(), ld_space]
        else:
            raise AssertionError

    @property
    def state_space(self):
        return NotImplemented

    def get_action_space(self):
        shape = (self._base_env.action_dim(), )
        return gym.spaces.Box(low=- np.ones(shape),
                              high=np.ones(shape),
                              shape=shape,
                              dtype=float)

    @property
    def max_seq_length(self) -> int:
        return 1000 // self._action_repeat

    @property
    def action_repeat(self):
        return self._action_repeat

    def render(self,
               img_size: Optional[tuple[int, int]] = None,
               camera_name: Optional[str] = None) -> np.ndarray:
        if img_size is None:
            img_size = self._img_size
        # base env takes care of default cam if camera_name=None
        return self._base_env.render(img_size=img_size, cam=camera_name)


    def get_background_mask(self, obs):
        return np.logical_and(obs[:, :, 2] > obs[:, :, 1], (obs[:, :, 2] > obs[:, :, 0]))

    def get_brightness(self) -> float:
        return self._base_env.get_brightness()

    def get_current_obs(self, obs_type: ObsTypes = None):
        return self._get_obs(state=self._current_state, obs_type=obs_type)

    @property
    def img_size(self):
        return self._img_size