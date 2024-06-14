import numpy as np

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations


class NoiseSource(AbstractDistractionSource):

    def __init__(self,
                 seed: int,
                 shape: tuple[int, int],
                 strength: int = 255,
                 intensity: int = 1):
        super().__init__(seed=seed)
        self.strength = strength
        self.shape = shape
        self.intensity = intensity
        self.mask = np.ones((self.shape[0], self.shape[1]))

    def get_info(self):
        info = super().get_info()
        info["strength"] = self.strength
        return info

    def get_image(self):
        w, h = self.shape
        img = self._rng.rand(w, h, 3) * self.strength
        img = img.astype(np.uint8)
        return img, self.mask

    def ground_allowed(self, ground) -> bool:
        return ground == DistractionLocations.BACKGROUND
