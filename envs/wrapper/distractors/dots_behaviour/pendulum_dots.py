import numpy as np

from envs.wrapper.distractors.dots_behaviour.abstract_dots import DotsBehaviour, Limits


class PendulumDotsSource(DotsBehaviour):
    def __init__(
        self,
        dt=0.05,
        sim_dt=1e-4,
        mass=1.0,
        g=9.81,
        friction=0.0,
        scale_factor=0.8,
        transition_noise_std=0.0,
    ):
        self.dt = dt
        self.sim_dt = sim_dt
        self.mass = mass
        self.g = g
        self.friction = friction
        self.scale_factor = scale_factor
        self.transition_noise_std = transition_noise_std

    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> dict:
        return {
            "pendulum_angle": np_random.uniform(-np.pi, np.pi, size=(num_dots)),
            "pendulum_velocity": np.zeros(num_dots),
            "lengths": np.ones(num_dots),
            "start_positions": np.concatenate(
                [
                    np_random.uniform(*x_lim, size=(num_dots, 1)),
                    np_random.uniform(*y_lim, size=(num_dots, 1)),
                ],
                axis=1,
            ),
            "rotation": np_random.choice([0, np.pi], size=(num_dots)),
        }

    def update_state(self, state: dict) -> dict:
        return {
            **state,
            **self._transition_function(state),
        }

    def get_positions(self, state: dict) -> np.array:
        positions = state["start_positions"].copy()
        positions[:, 0] += 0.5 * np.sin(state["pendulum_angle"] + state["rotation"])
        positions[:, 1] += 0.5 * np.cos(state["pendulum_angle"] + state["rotation"])
        return positions

    def _transition_function(self, state: dict) -> dict:
        nSteps = self.dt / self.sim_dt
        if nSteps != np.round(nSteps):
            print("Warning from Pendulum: dt does not match up")
            nSteps = np.round(nSteps)

        inertia = self.mass * (state["lengths"] ** 2) / 3
        c = self.g * state["lengths"] * self.mass / inertia
        for i in range(0, int(nSteps)):
            velNew = state["pendulum_velocity"] + self.sim_dt * (
                c * np.sin(state["pendulum_angle"])
                - state["pendulum_velocity"] * self.friction
            )
            state["pendulum_angle"] = state["pendulum_angle"] + self.sim_dt * velNew
            state["pendulum_velocity"] = velNew
        if self.transition_noise_std > 0.0:
            state["pendulum_velocity"] += self.random.normal(
                loc=0.0, scale=self.transition_noise_std
            )
        return state
