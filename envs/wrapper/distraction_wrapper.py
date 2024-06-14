import numpy as np
import gym

from typing import Optional
import cv2
from envs.wrapper.merge_strategy.merge_strategies import (DistractionLocations,
                                                          FrontMerge,
                                                          BackgroundMerge,
                                                          FrontAndBackMerge)

from envs.common.abstract_mbrl_env import AbstractMBRLEnv
from envs.wrapper.distractors.abstract_source import DistractionSources
from envs.wrapper.distractors.noise import NoiseSource
from envs.wrapper.distractors.random_color import RandomColorSource
from envs.wrapper.distractors.pre_rendered_source import PreRenderedSource
from envs.wrapper.distractors.dots_source import DotsSource
from envs.wrapper.distractors.davis_source import DAVISDataSource
from envs.wrapper.distractors.kinetics400_source import Kinetics400DataSource


class DistractionWrapper(gym.Wrapper):

    def __init__(self,
                 env: AbstractMBRLEnv,
                 distraction_type: DistractionSources,
                 seed: int,
                 img_idx: int = 0,
                 img_size: Optional[tuple[int, int]] = None,
                 ground=None,
                 difficulty=None,
                 folder: Optional[str] = None,
                 mask_to_info: bool = False):
        super(DistractionWrapper, self).__init__(env=env)

        self._img_idx = img_idx

        difficulty = "hard" if difficulty is None else difficulty
        if any(self.env.obs_are_images):
            shape2d = (self.env.observation_space[img_idx].shape[0], self.env.observation_space[img_idx].shape[1])
        else:
            shape2d = self.env.img_size
        if isinstance(distraction_type, str):
            if distraction_type == DistractionSources.COLOR:
                self._distraction_source = RandomColorSource(seed=seed, shape=shape2d)
            elif distraction_type == DistractionSources.NOISE:
                self._distraction_source = NoiseSource(seed=seed, shape=shape2d)
            elif distraction_type in [DistractionSources.DISKS_EASY,
                                      DistractionSources.DISKS_MEDIUM,
                                      DistractionSources.DISKS_HARD,
                                      DistractionSources.WALLS]:
                assert folder is not None, "Folder must be specified for pre-rendered distractors"
                temp_step = 1 if distraction_type == DistractionSources.WALLS else 5
                self._distraction_source = PreRenderedSource(seed=seed,
                                                             shape=shape2d,
                                                             vid_folder=folder,
                                                             max_seq_length=self.env.max_seq_length,
                                                             temp_step=temp_step)
            elif "dots_" in distraction_type:
                dots_behaviour = distraction_type.split("_")[1]
                self._distraction_source = DotsSource(seed=seed,
                                                      shape=shape2d,
                                                      dots_behaviour=dots_behaviour)
            elif distraction_type == DistractionSources.DAVIS:
                assert folder is not None
                self._distraction_source = DAVISDataSource(seed=seed,
                                                           shape=shape2d,
                                                           difficulty=difficulty,
                                                           data_path=folder)
            elif distraction_type in [DistractionSources.KINETICS, DistractionSources.KINETICS_GRAY]:
                assert folder is not None
                grayscale = distraction_type == DistractionSources.KINETICS_GRAY
                self._distraction_source = Kinetics400DataSource(seed=seed,
                                                                 shape=shape2d,
                                                                 difficulty=difficulty,
                                                                 grayscale=grayscale,
                                                                 data_path=folder,
                                                                 seq_length=self.env.max_seq_length + 1)
            else:
                raise Exception(
                    f"Distractor of type {distraction_type} not known. Please choose a distractor type from "
                    f"distractor type enum."
                )

        else:
            # Given class
            self._distraction_source = distraction_type(shape2d)
        assert self._distraction_source.ground_allowed(ground), \
            "location \'{}\' not available for type \'{}\'".format(ground, distraction_type)

        if ground == DistractionLocations.FOREGROUND:
            self.merger = FrontMerge(self._distraction_source)
        elif ground == DistractionLocations.BACKGROUND:
            self.merger = BackgroundMerge(self._distraction_source)
        elif ground == DistractionLocations.BOTH:
            self.merger = FrontAndBackMerge(self._distraction_source)
        else:
            raise AssertionError
        self._img_size = img_size
        self._resize = self.will_resize(distraction_type=distraction_type)
        if self._resize and self.env.obs_are_images[self._img_idx]:
            assert self._img_size is not None
            spaces = list(self.env.observation_space)
            spaces[img_idx] = gym.spaces.Box(low=0,
                                             high=255,
                                             shape=(self._img_size[0], self._img_size[1], 3),
                                             dtype=np.uint8)
            self.observation_space = gym.spaces.Tuple(spaces)

        self._mask_to_info = mask_to_info

    def reset(self, eval_mode: bool = False, *args, **kwargs) -> tuple[list[np.ndarray], dict]:
        obs, info = self.env.reset(eval_mode, *args, **kwargs)
        self._distraction_source.reset(eval_mode=eval_mode)
        if self.env.obs_are_images[self._img_idx]:
            obs[self._img_idx] = self.merger.merge(obs=obs[self._img_idx],
                                                   background_mask=self.env.get_background_mask(obs=obs[self._img_idx]))
            if self._resize:
                obs[self._img_idx] = self.resize(obs[self._img_idx])
                return obs, info
            else:
                piv = [None] * len(obs)
                piv[self._img_idx] = np.logical_not(self.merger.get_last_mask())
                return obs, (info | {"pixel_is_valid": piv}) if self._mask_to_info else info
        else:
            return obs, info

    def step(self, action: np.ndarray) -> tuple[list[np.ndarray], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.obs_are_images[self._img_idx]:
            obs[self._img_idx] = self.merger.merge(obs=obs[self._img_idx],
                                                   background_mask=self.env.get_background_mask(obs=obs[self._img_idx]))
            if self._resize:
                obs[self._img_idx] = self.resize(obs[self._img_idx])
                return obs, reward, terminated, truncated, info
            else:
                piv = [None] * len(obs)
                piv[self._img_idx] = np.logical_not(self.merger.get_last_mask())
                return obs, reward, terminated, truncated, \
                    (info | {"pixel_is_valid": piv}) if self._mask_to_info else info
        else:
            return obs, reward, terminated, truncated, info

    def resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self._img_size[1], self._img_size[0]))

    @staticmethod
    def will_resize(distraction_type: str) -> bool:
        return distraction_type in [DistractionSources.DAVIS,
                                    DistractionSources.KINETICS,
                                    DistractionSources.KINETICS_GRAY]

    def get_current_obs(self, obs_type=None, return_original_image: bool = False):
        original_image = None
        new_obs = self.env.get_current_obs(obs_type)
        if self.env.get_obs_are_images(obs_type)[self._img_idx]:
            if return_original_image:
                original_image = new_obs[self._img_idx].copy()
            new_obs[self._img_idx] = \
                self.merger.merge(obs=new_obs[self._img_idx],
                                  background_mask=self.env.get_background_mask(obs=new_obs[self._img_idx]))
            if self._resize:
                new_obs[self._img_idx] = self.resize(new_obs[self._img_idx])
        #plt.figure("plt")
        #plt.clf()
        #plt.imshow(new_obs[self._img_idx])
        #plt.pause(0.001)
        if return_original_image:
            return new_obs, original_image
        else:
            return new_obs
