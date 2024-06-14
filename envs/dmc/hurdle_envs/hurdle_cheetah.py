import os
import numpy as np
from typing import Optional

from envs.dmc.abstract_base_env import AbstractBaseEnv

import matplotlib
import matplotlib.cm

class HurdleCheetah(AbstractBaseEnv):

    _MAX_NUM_OBSTACLES = 100

    _RUN_SPEED = 10

    _DEFAULT_TIME_LIMIT = 10

    _AWAY_POS = np.array([-90.0, 0.0, 0.0])

    _DIR = os.path.dirname(__file__)
    _FILENAME = "assets/hurdle_cheetah.xml"

    _JOINT_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

    _STATE_SIZE = 17
    _NUM_JOINTS = 6
    _POSITION_SIZE = 8
    _DEFAULT_ACTION_REPEAT = 4

    _valid_proprioception_modes = ["joints_only", "joints_velo", "joints_velo_rot"]

    def __init__(self,
                 task: str,
                 seed: int,
                 min_obstacle_height: float = 0.075,
                 max_obstacle_height: float = 0.15,
                 cam: str = "egocentric",
                 proprioception_mode: str = "joints_velo"):
        from dm_control.rl import control
        from dm_control.suite import cheetah

        assert task == "run"
        assert proprioception_mode in self._valid_proprioception_modes
        print("Using proprioception mode: {}".format(proprioception_mode))

        self._rng = np.random.RandomState(seed=seed)

        self._min_obstacle_height = min_obstacle_height
        self._max_obstacle_height = max_obstacle_height
        self._min_obstacle_distance = 2.0
        self._max_obstacle_distance = 200.0

        self._default_cam = cam
        self._proprioception_mode = proprioception_mode
        self._physics = cheetah.Physics.from_xml_path(os.path.join(HurdleCheetah._DIR, HurdleCheetah._FILENAME))

        task = cheetah.Cheetah(random=self._rng)
        self._env = control.Environment(physics=self._physics,
                                        task=task,
                                        time_limit=HurdleCheetah._DEFAULT_TIME_LIMIT)

        self._cmap = matplotlib.cm.get_cmap('autumn')

    def _shuffle_obstacles(self):

        for i in range(self._MAX_NUM_OBSTACLES):
            obstacle = "box_{:02d}".format(i)

            box_size = [0.1, 0.8, self._rng.uniform(low=self._min_obstacle_height, high=self._max_obstacle_height)]
            height_offset = np.array([0, 0, box_size[2]])
            random_pos = self._rng.uniform([self._min_obstacle_distance, 0.0, 0.0],
                                           [self._max_obstacle_distance, 0.0, 0.0])

            self._physics.named.model.geom_size[obstacle] = box_size
            self._physics.named.model.geom_rgba[obstacle] = self._cmap(self._rng.uniform(0.0, 1.0))
            self._physics.named.model.geom_pos[obstacle] = random_pos + height_offset

    def _get_joint_pos(self) -> np.ndarray:
        return self._physics.named.data.qpos[HurdleCheetah._JOINT_NAMES]

    def _get_joint_vel(self) -> np.ndarray:
        return self._physics.named.data.qvel[HurdleCheetah._JOINT_NAMES]

    def render(self, img_size: tuple[int, int], cam: Optional[str] = None) -> np.ndarray:
        cam = "back" if cam is not None and cam == "default" else cam
        img = self._env.physics.render(camera_id=self._default_cam if cam is None else cam,
                                       height=img_size[0],
                                       width=img_size[1])
        return img

    def reset(self):
        self._shuffle_obstacles()
        state = self._env.reset()
        state.observation["x_pos"] = self._env.physics.named.data.xpos["torso", "x"]
        return state

    def step(self, action: np.ndarray):
        state = self._env.step(action)
        state.observation["x_pos"] = self._env.physics.named.data.xpos["torso", "x"]
        return state

    def get_proprioceptive_position(self, state) -> np.ndarray:
        return state.observation["position"][-(7 if self._proprioception_mode == "joints_velo_rot" else 6):]

    def get_proprioceptive_velocity(self, state) -> np.ndarray:
        return state.observation["velocity"][-(6 if self._proprioception_mode == "joints_only" else 9):]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]

    @staticmethod
    def get_state(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["velocity"]], axis=-1)

    @property
    def proprioceptive_pos_size(self):
        return 7 if self._proprioception_mode == "joints_velo_rot" else 6

    @property
    def proprioceptive_vel_size(self):
        return 6 if self._proprioception_mode == "joints_only" else 9

    @property
    def position_size(self):
        return self._POSITION_SIZE

    @property
    def state_size(self):
        return self._STATE_SIZE

    @property
    def default_action_repeat(self) -> int:
        return self._DEFAULT_ACTION_REPEAT

    @property
    def physics(self):
        return self._physics

    @staticmethod
    def get_info(state):
        return {"x_pos": state.observation["x_pos"]}

    def action_dim(self):
        return self._env.action_spec().shape[0]

    def observation_spec(self):
        return self._env.observation_spec()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dm_control import viewer

    env = HurdleCheetah(task="run",
                        seed=0)

    for i in range(5):
        env.reset()

    viewer.launch(env)

    img_ego = env.render(img_size=(480, 480))
    img_ext = env.render(img_size=(480, 480), cam="back")

    plt.figure()
    plt.imshow(img_ego)
    plt.axis("off")
    plt.figure()
    plt.imshow(img_ext)
    plt.axis("off")
    plt.show()

