import numpy as np

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations


class RandomColorSource(AbstractDistractionSource):

    def __init__(self,
                 seed: int,
                 shape: tuple[int, int],
                 intensity: int = 1):
        super().__init__(seed=seed)
        self.shape = shape
        self.intensity = intensity
        self.bg = np.zeros((self.shape[0], self.shape[1], 3))
        self.mask = np.ones((self.shape[0], self.shape[1]))
        self._color = None
        self.reset()

    def reset(self, eval_mode: bool = False):
        self._color = self._rng.randint(0, 256, size=(3,))
        self.bg[:, :] = self._color

    def get_info(self):
        info = super().get_info()
        info["color"] = self._color
        return info

    def get_image(self):
        return self.bg, self.mask

    def ground_allowed(self, ground):
        return ground == DistractionLocations.BACKGROUND
