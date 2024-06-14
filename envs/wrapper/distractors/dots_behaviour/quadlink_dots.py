import numpy as np

from envs.wrapper.distractors.dots_behaviour.abstract_dots import DotsBehaviour, Limits


class QuadLinkState:
    angles: np.array
    velocities: np.array
    lengths: np.array
    start_positions: np.array


class QuadLinkDotsSource(DotsBehaviour):
    def __init__(
        self,
        dt=0.05,
        sim_dt=1e-4,
        masses=None,
        g=9.81,
        friction=None,
        scale_factor=None,
    ):
        import n_link_sim as sim
        self._sim = sim
        self.dt = dt
        self.sim_dt = sim_dt
        self.masses = masses if masses is not None else np.ones(4)
        self.g = g
        self.friction = friction if friction is not None else np.zeros(4)
        self.scale_factor = scale_factor


    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> QuadLinkState:
        return {
            "angles": np_random.uniform(-np.pi, np.pi, size=(num_dots, 4)),
            "velocities": np.zeros((num_dots, 4)),
            "lengths": np.ones((num_dots, 4)),
            "start_positions": np.concatenate(
                [
                    np_random.uniform(*x_lim, size=(num_dots, 1)),
                    np_random.uniform(*y_lim, size=(num_dots, 1)),
                ],
                axis=1,
            ),
        }

    def update_state(self, state: QuadLinkState) -> QuadLinkState:
        return self._transition_function(state)

    def _transition_function(self, state: QuadLinkState) -> QuadLinkState:
        quadlink_state = np.empty(
            (
                state["angles"].shape[0],
                state["angles"].shape[1] + state["velocities"].shape[1],
            ),
            dtype=state["angles"].dtype,
        )
        quadlink_state[:, 0::2] = state["angles"]
        quadlink_state[:, 1::2] = state["velocities"]

        intertias = self.masses * (state["lengths"] ** 2) / 3

        actions = np.zeros((quadlink_state.shape[0], 4))
        result = self._sim.simulate_quad_link(
            states=quadlink_state,
            actions=actions,
            dt=self.dt,
            masses=self.masses,
            lengths=state["lengths"],
            inertias=intertias,
            g=self.g,
            friction=self.friction,
            dst=self.sim_dt,
        )

        new_angles = result[:, 0:8:2]
        new_velocities = result[:, 1:8:2]

        return {**state, "angles": new_angles, "velocities": new_velocities}

    def get_positions(self, state: dict) -> np.array:
        positions = state["start_positions"].copy()
        for i in range(4):
            positions[:, 0] += (
                0.125 * np.sin(state["angles"][:, i]) * state["lengths"][:, i]
            )
            positions[:, 1] += (
                0.125 * np.cos(state["angles"][:, i]) * state["lengths"][:, i]
            )
        return positions
