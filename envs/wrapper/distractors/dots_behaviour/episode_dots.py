import numpy as np

from envs.wrapper.distractors.dots_behaviour.abstract_dots import DotsBehaviour, Limits


class EpisodeDotsSource(DotsBehaviour):
    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> dict:
        return {
            "positions": np.concatenate(
                [
                    np_random.uniform(*x_lim, size=(num_dots, 1)),
                    np_random.uniform(*y_lim, size=(num_dots, 1)),
                ],
                axis=1,
            ),
        }

    def update_state(self, state: dict) -> dict:
        return state

    def get_positions(self, state: dict) -> np.array:
        return state["positions"]
