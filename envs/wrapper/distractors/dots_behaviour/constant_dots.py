import numpy as np

from envs.wrapper.distractors.dots_behaviour.abstract_dots import DotsBehaviour, Limits


class ConstantDots(DotsBehaviour):
    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> dict:
        return {
            # Fix always yield the same
            "positions": np.stack(
                [np.linspace(*x_lim, num=num_dots), np.linspace(*y_lim, num=num_dots)],
                axis=1,
            )
        }

    def update_state(self, state: dict) -> dict:
        return state

    def get_positions(self, state: dict) -> np.array:
        return state["positions"]
