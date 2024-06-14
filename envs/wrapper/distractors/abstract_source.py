from abc import ABCMeta, abstractmethod
from enum import Enum, EnumMeta
import numpy as np


class EnumContainsMeta(EnumMeta):
    def __contains__(self, item):
        return item in self.__members__.values()


class DistractionSources(str, Enum, metaclass=EnumContainsMeta):
    NOISE = "noise"
    COLOR = "color"
    DISKS_EASY = "disks_pre_rendered_easy"
    DISKS_MEDIUM = "disks_pre_rendered_medium"
    DISKS_HARD = "disks_pre_rendered_hard"
    WALLS = "walls_pre_rendered"
    DAVIS = "davis"
    KINETICS = "kinetics400"
    KINETICS_GRAY = "kinetics400_gray"

    DOTS_CONSTANT = "dots_constant"
    DOTS_EPISODE = "dots_episode"
    DOTS_LINEAR = "dots_linear"
    DOTS_PENDULUM = "dots_pendulum"
    DOTS_QUADLINK = "dots_quadlink"
    DOTS_RANDOM = "dots_random"


class AbstractDistractionSource(object, metaclass=ABCMeta):

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def get_image(self) -> tuple[np.array, np.array]:
        pass

    def reset(self, eval_mode: bool = False):
        pass

    def get_info(self) -> dict:
        info = {}
        return info

    def ground_allowed(self, ground) -> bool:
        raise NotImplementedError

    @property
    def ignore_mask_for_background(self) -> bool:
        return False
