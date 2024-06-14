from typing import NamedTuple
import cv2
import numpy as np

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource
from envs.wrapper.distractors.dots_behaviour.constant_dots import ConstantDots
from envs.wrapper.distractors.dots_behaviour.episode_dots import EpisodeDotsSource
from envs.wrapper.distractors.dots_behaviour.random_dots import RandomDotsSource
from envs.wrapper.distractors.dots_behaviour.linear_dots import LinearDotsSource
from envs.wrapper.distractors.dots_behaviour.pendulum_dots import PendulumDotsSource
from envs.wrapper.distractors.dots_behaviour.quadlink_dots import QuadLinkDotsSource
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations


class Limits(NamedTuple):
    low: float
    high: float


class DotsSource(AbstractDistractionSource):

    BEHAVIOURS = {"constant": ConstantDots,
                  "episode": EpisodeDotsSource,
                  "random": RandomDotsSource,
                  "linear": LinearDotsSource,
                  "pendulum": PendulumDotsSource,
                  "quadlink": QuadLinkDotsSource}

    def __init__(self,
                 seed: int,
                 shape: tuple[int, int],
                 dots_behaviour: str,
                 dots_size=0.12):
        super().__init__(seed=seed)
        self.shape = shape
        self.num_dots = 12
        self.dots_size = dots_size
        self.x_lim = Limits(0.05, 0.95)
        self.y_lim = Limits(0.05, 0.95)

        self.dots_behaviour = self.BEHAVIOURS[dots_behaviour]()
        self.dots_state = self.dots_behaviour.init_state(self.num_dots, self.x_lim, self.y_lim, self._rng)
        self.positions = self.dots_behaviour.get_positions(self.dots_state)
        self.dots_parameters = self.init_dots()

    def get_info(self):
        info = super().get_info()
        return {
            **info,
            "num_dots": self.num_dots,
            "size": self.dots_size,
        }

    def init_dots(self) -> dict:
        return {
            "colors": self._rng.random((self.num_dots, 3)),
            "sizes": self._rng.uniform(0.8, 1.2, size=(self.num_dots, 1)),
        }

    def reset(self, eval_mode: bool, seed=None):
        super().reset(seed)
        self.dots_parameters = self.init_dots()
        self.dots_state = self.dots_behaviour.init_state(
            self.num_dots, self.x_lim, self.y_lim, self._rng
        )

    def build_bg(self, w, h):
        bg = np.zeros((h, w, 3))
        positions = self.dots_behaviour.get_positions(self.dots_state) * [[w, h]]
        sizes = self.dots_parameters["sizes"]
        colors = self.dots_parameters["colors"]
        for position, size, color in zip(positions, sizes, colors):
            cv2.circle(
                bg,
                (int(position[0]), int(position[1])),
                int(size * w * self.dots_size),
                color,
                -1,
            )

        self.dots_state = self.dots_behaviour.update_state(self.dots_state)
        bg *= 255
        return bg.astype(np.uint8)

    def get_image(self):
        h, w = self.shape
        img = self.build_bg(w, h)
        mask = np.logical_or(img[:, :, 0] > 0, img[:, :, 1] > 0, img[:, :, 2] > 0)
        return img, mask

    def ground_allowed(self, ground) -> bool:
        return ground in [DistractionLocations.FOREGROUND, DistractionLocations.BACKGROUND]