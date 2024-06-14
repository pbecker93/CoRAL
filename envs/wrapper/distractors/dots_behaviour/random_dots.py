import numpy as np

from envs.wrapper.distractors.dots_behaviour.abstract_dots import DotsBehaviour, Limits


class RandomDotsSource(DotsBehaviour):
    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> dict:
        return {
            "num_dots": num_dots,
            "x_lim": x_lim,
            "y_lim": y_lim,
            "np_random": np_random,
        }

    def update_state(self, state: dict) -> dict:
        return state

    def get_positions(self, state: dict) -> np.array:
        np_random, x_lim, y_lim, num_dots = (
            state["np_random"],
            state["x_lim"],
            state["y_lim"],
            state["num_dots"],
        )
        return np.concatenate(
            [
                np_random.uniform(*x_lim, size=(num_dots, 1)),
                np_random.uniform(*y_lim, size=(num_dots, 1)),
            ],
            axis=1,
        )
