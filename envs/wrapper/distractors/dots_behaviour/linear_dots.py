from enum import IntEnum

import numpy as np

from envs.wrapper.distractors.dots_behaviour.abstract_dots import DotsBehaviour, Limits


class GravitationalConstant(IntEnum):
    PLANET = 1
    ELECTRONS = -1
    IDEAL_GAS = 0


class LinearDotsSource(DotsBehaviour):
    def __init__(
        self,
        base_velocity: float = 0.5,
        gravitation: GravitationalConstant = GravitationalConstant.IDEAL_GAS,
    ):
        self.v = base_velocity
        self.gravitation = gravitation

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
            "velocities": (np_random.normal(0, 0.01, size=(num_dots, 2)) * self.v),
            "sizes": np_random.uniform(0.8, 1.2, size=(num_dots, 1)),
            "x_lim": x_lim,
            "y_lim": y_lim,
        }

    def update_state(self, state: dict) -> dict:
        def compute_acceleration(
            positions: np.array, sizes: np.array, gravitation: int
        ):
            accelerations = np.zeros(positions.shape)
            for i in range(len(positions)):
                relative_positions = positions - positions[i]
                distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
                distances[i] = 1

                force_vectors = (
                    relative_positions * gravitation * (sizes**2) / (distances**2)
                )
                accelerations[i] = 0.00001 * np.sum(force_vectors, axis=0)

            return accelerations

        accelerations = compute_acceleration(
            state["positions"], state["sizes"], self.gravitation
        )
        velocities = state["velocities"] + accelerations
        positions = state["positions"] + velocities

        new_state = {
            **state,
            "positions": positions,
            "velocities": velocities,
        }

        new_state = self._limit_positions(
            new_state, x_lim=state["x_lim"], y_lim=state["y_lim"]
        )
        return new_state

    def get_positions(self, state: dict) -> np.array:
        return state["positions"]

    def _limit_positions(self, state: dict, x_lim: Limits, y_lim: Limits) -> dict:
        for i in range(state["positions"].shape[0]):
            if not x_lim.high >= state["positions"][i][0] >= x_lim.low:
                state["velocities"][i][0] = -state["velocities"][i][0]
            if not y_lim.high >= state["positions"][i][1] >= y_lim.low:
                state["velocities"][i][1] = -state["velocities"][i][1]

        return state
