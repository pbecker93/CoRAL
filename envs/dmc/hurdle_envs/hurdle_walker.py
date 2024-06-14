import os
import numpy as np
from typing import Optional

from envs.dmc.abstract_base_env import AbstractBaseEnv

import matplotlib
import matplotlib.cm

class HurdleWalker(AbstractBaseEnv):

    _MAX_NUM_OBSTACLES = 100

    _SPEEDS = {"walk": 1,
               "run": 8}

    #_MAX_BOX_DIST_FACTOR = 30

    _DEFAULT_TIME_LIMIT = 25
    _CONTROL_TIMESTEP = .025

    _AWAY_POS = np.array([-250.0, 0.0, 0.0])

    _DIR = os.path.dirname(__file__)
    _FILENAME = "assets/hurdle_walker.xml"

    _JOINT_NAMES = ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]

    _PROPRIOCEPTIVE_POS_SIZE = 14
    _PROPRIOCEPTIVE_VEL_SIZE = 6
    _NUM_JOINTS = 6
    _POSITION_SIZE = 15
    _DEFAULT_ACTION_REPEAT = 2

    def __init__(self,
                 task: str,
                 seed: int,
                 min_obstacle_height: float = 0.075,
                 max_obstacle_height: float = 0.15,
                 cam: str = "egocentric"):
        from dm_control.rl import control
        from dm_control.suite import walker

        assert task in HurdleWalker._SPEEDS.keys(), "Unknown task, use any either 'walk' or 'run'"

        self._rng = np.random.RandomState(seed=seed)

        self._min_obstacle_height = min_obstacle_height
        self._max_obstacle_height = max_obstacle_height
        self._min_obstacle_distance = 2.0
        self._max_obstacle_distance = 200.0

        self._default_cam = cam
        self._physics = walker.Physics.from_xml_path(os.path.join(HurdleWalker._DIR, HurdleWalker._FILENAME))

        task = walker.PlanarWalker(move_speed=HurdleWalker._SPEEDS[task], random=self._rng)
        self._env = control.Environment(physics=self._physics,
                                        task=task,
                                        time_limit=HurdleWalker._DEFAULT_TIME_LIMIT,
                                        control_timestep=HurdleWalker._CONTROL_TIMESTEP)

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
        return self._physics.named.data.qpos[HurdleWalker._JOINT_NAMES]

    def _get_joint_vel(self) -> np.ndarray:
        return self._physics.named.data.qvel[HurdleWalker._JOINT_NAMES]

    def render(self, img_size: tuple[int, int], cam: Optional[str] = None) -> np.ndarray:
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

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["orientations"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][3:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        np_state = np.concatenate([state.observation["orientations"], np.array([state.observation["height"]])])
        return np_state.astype(np.float32)

    @property
    def proprioceptive_pos_size(self):
        return self._PROPRIOCEPTIVE_POS_SIZE

    @property
    def proprioceptive_vel_size(self):
        return self._PROPRIOCEPTIVE_VEL_SIZE

    @property
    def position_size(self):
        return self._POSITION_SIZE

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
