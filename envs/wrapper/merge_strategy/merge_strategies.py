from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import Enum, EnumMeta
import numpy as np
import time

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource


class EnumContainsMeta(EnumMeta):
    def __contains__(self, item):
        return item in self.__members__.values()


class DistractionLocations(str, Enum, metaclass=EnumContainsMeta):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    BOTH = "both"


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, source: AbstractDistractionSource, intensity=1):
        self.source = source
        self.intensity = intensity

    @abstractmethod
    def merge(self, obs: np.array, background_mask: np.array) -> np.array:
        pass

    def merge_timeseries(self, obs: np.array, background_mask: np.array) -> np.array:
        """
        Used for offline adding of observations
        :param obs:
        :return:
        """
        self.source.reset()
        augmented_obs = []
        for timestep, mask in zip(obs, background_mask):
            augmented_obs.append(self.merge(timestep, mask))
        return np.array(augmented_obs)

    @abstractmethod
    def get_last_mask(self):
        pass


class FrontMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array, background_mask: np.array) -> np.array:
        img, mask = self.source.get_image()
        augmented_obs = np.copy(obs)
        augmented_obs[mask] = img[mask]
        self._mask = mask
        return augmented_obs

    def get_last_mask(self):
        return self._mask


class BackgroundMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array, background_mask: np.array) -> np.array:
        if background_mask is None:
            raise NotImplementedError("Background mask is None")
        img, mask = self.source.get_image()
        if not self.source.ignore_mask_for_background:
            combined_mask = np.logical_and(mask, background_mask)
        else:
            combined_mask = background_mask
        obs = np.where(combined_mask[..., None], img, obs)
        self._mask = combined_mask
        return obs

    def get_last_mask(self):
        return self._mask


class FrontAndBackMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array, background_mask: np.array) -> np.array:
        if background_mask is None:
            raise NotImplementedError("Background mask is None")
        img, mask = self.source.get_image()
        background_mask = np.logical_and(~mask, background_mask)
        augmented_obs = np.copy(obs)
        # background
        augmented_obs[background_mask] = img[background_mask]
        # foreground
        augmented_obs[mask] = img[mask]

        self._mask = np.logical_or(background_mask, mask)
        return augmented_obs

    def get_last_mask(self):
        return self._mask
