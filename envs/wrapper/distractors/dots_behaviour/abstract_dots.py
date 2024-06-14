import numpy as np
from typing import NamedTuple


class Limits(NamedTuple):
    low: float
    high: float


class DotsBehaviour:

    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> dict:
        pass

    def update_state(self, state):
        raise NotImplementedError

    def get_positions(self, state) -> np.array:
        raise NotImplementedError
