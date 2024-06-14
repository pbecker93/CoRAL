from typing import Optional
import numpy as np


from envs.dmc.abstract_base_env import AbstractBaseEnv


class _AntCorridorEnv(AbstractBaseEnv):

    _CONTROL_TIME_STEP = 0.025

    _POSITION_SIZE = 8
    _CORRIDOR_LENGTH = 80
    DEFAULT_ACTION_REPEAT = 2

    _valid_proprioception_modes = ["joints_only", "joints_sensors"]

    def __init__(self,
                 seed: int,
                 cam: str = "egocentric",
                 proprioception_mode: str = "joints_sensors"):

        from dm_control import composer
        from dm_control.locomotion import tasks, walkers

        assert proprioception_mode in self._valid_proprioception_modes
        print("Using proprioception mode: {}".format(proprioception_mode))

        self._default_cam = cam

        agent = walkers.Ant()

        arena = self._build_arena()

        task = tasks.RunThroughCorridor(walker=agent,
                                        arena=arena,
                                        contact_termination=None,
                                        terminate_at_height=None,
                                        control_timestep=_AntCorridorEnv._CONTROL_TIME_STEP)
        self._env = composer.Environment(time_limit=_AntCorridorEnv._CONTROL_TIME_STEP * 999,
                                         task=task,
                                         random_state=seed,
                                         strip_singleton_obs_buffer_dim=True)

        self.action_spec = self._env.action_spec
        self.observation_spec = self._env.observation_spec

        self._proprioception_mode = proprioception_mode

    def reset(self):
        state = self._env.reset()
        state.observation["walker/x_pos"] = self._env.physics.named.data.xpos["walker/", "x"]
        return state

    def step(self, action: np.ndarray):
        state = self._env.step(action)
        state.observation["walker/x_pos"] = self._env.physics.named.data.xpos["walker/", "x"]
        return state

    def _build_arena(self):
        raise NotImplementedError

    def render(self, img_size: tuple[int, int], cam: Optional[str] = None) -> np.ndarray:
        cam = "floating" if cam is not None and cam == "default" else cam
        img = self._env.physics.render(camera_id="walker/{}".format(self._default_cam if cam is None else cam),
                                       height=img_size[0],
                                       width=img_size[1])
        return img

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["walker/joints_pos"]

    def get_proprioceptive_velocity(self, state) -> np.ndarray:
        if self._proprioception_mode == "joints_only":
            return state.observation["walker/joints_vel"]
        elif self._proprioception_mode == "joints_sensors":
            return np.concatenate([state.observation["walker/joints_vel"],
                                   state.observation["walker/sensors_gyro"],
                                   state.observation["walker/sensors_velocimeter"],
                                   state.observation["walker/sensors_touch"]], axis=-1)

        raise NotImplementedError

    @staticmethod
    def get_position(state) -> np.ndarray:
        np_state = np.concatenate([state.observation["walker/joints_pos"],
                                   state.observation["walker/height"],
                                   state.observation["walker/bodies_pos"],
                                   state.observation["walker/bodies_quad"]])
        return np_state.astype(np.float32)

    @property
    def proprioceptive_pos_size(self):
        return 8

    @property
    def proprioceptive_vel_size(self):
        return 8 + (0 if self._proprioception_mode == "joints_only" else 15)

    @property
    def position_size(self):
        return self._POSITION_SIZE

    @staticmethod
    def get_info(state):
        return {"x_pos": state.observation["walker/x_pos"]}

    @property
    def default_action_repeat(self) -> int:
        return self.DEFAULT_ACTION_REPEAT

    def action_spec(self):
        return self._env.action_spec

    def observation_spec(self):
        return self._env.observation_spec

    @property
    def physics(self):
        return self._env.physics

    def action_dim(self):
        return self.action_spec().shape[0]


class AntEmptyCorridorEnv(_AntCorridorEnv):

    def _build_arena(self):
        from envs.dmc.corridor_ant.custom_corridors import EmptyCorridor
        return EmptyCorridor(corridor_length=self._CORRIDOR_LENGTH)


class AntWallsCorridorEnv(_AntCorridorEnv):

    _MIN_GAP = 2
    _MAX_GAP = 4
    _MIN_WIDTH = 1
    _MAX_WIDTH = 2

    def _build_arena(self):
        from dm_control.composer.variation import distributions
        from envs.dmc.corridor_ant.custom_corridors import WallsCorridor
        return WallsCorridor(corridor_length=self._CORRIDOR_LENGTH,
                             wall_gap=distributions.Uniform(low=self._MIN_GAP, high=self._MAX_GAP),
                             wall_width=distributions.Uniform(low=self._MIN_WIDTH, high=self._MAX_WIDTH),
                             wall_rgba=(0, 0, 1, 1),
                             visible_side_planes=True)


if __name__ == "__main__":

    from dm_control import viewer
    import matplotlib.pyplot as plt

    np.random.seed(0)

    env = AntWallsCorridorEnv()
    env.reset()
    env.reset()
    env.step(np.random.uniform(-1, 1, size=env.action_spec().shape))
    viewer.launch(env._env)

    _img = env.render(img_size=(64, 64))
    plt.figure()
    plt.imshow(_img)
    plt.axis("off")
    plt.show()
