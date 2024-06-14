import gymnasium as gym
import cv2
from typing import Optional, Union
import os

from mbrl_envs.common.abstract_mbrl_env import AbstractMBRLEnv
from mbrl_envs.common.obs_types import ObsTypes

from mbrl_envs.maniskill_envs.adapted_envs.replica_cad_envs import *
from mbrl_envs.maniskill_envs.adapted_envs.matterport import camera_poses as matterport_camera_poses
from mbrl_envs.maniskill_envs.adapted_envs.matterport import env_kwargs as matterport_env_kwargs

from mbrl_envs.maniskill_envs.adapted_envs.hacked_door_drawer import *


class ManiSkillMBRLEnv(AbstractMBRLEnv):
    SUPPORTED_OBS_TYPES = [ObsTypes.STATE,
                           ObsTypes.PROPRIOCEPTIVE,
                           ObsTypes.IMAGE,
                           ObsTypes.GRIPPER,
                           ObsTypes.IMAGE_PROPRIOCEPTIVE,
                           ObsTypes.GRIPPER_PROPRIOCEPTIVE,

                           ObsTypes.GRIPPER_GOAL,
                           ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL,

                           ObsTypes.DEPTH,
                           ObsTypes.SEGMENTATION,
                           ObsTypes.DEPTH_PROPRIOCEPTIVE,
                           ObsTypes.SEGMENTATION_PROPRIOCEPTIVE,
                           ObsTypes.IMG_SEG_PROPRIO,
                           ObsTypes.RGBD,
                           ObsTypes.RGBD_PROPRIOCEPTIVE,

                           ObsTypes.IMG_DEPTH_PROPRIO
                           ]
    SUPPORTED_TASKS = {
        "liftcube": "LiftCube-v0",
        "pickcube": "PickCube-v0",
        "plugcharger": "PlugCharger-v0",
        "peginsert": "PegInsertionSide-v0",
        "faucet": "HackedTurnFaucet-v0",

        "door": "OpenCabinetDoor-v1",
        "drawer": "OpenCabinetDrawer-v1",

        "hacked_door": "OpenHackedCabinetDoor-v1",
        "hacked_drawer": "OpenHackedCabinetDrawer-v1",

        "door_one_hot": "DoorOneHotGoal-v0",
        "drawer_one_hot": "DrawerOneHotGoal-v0",

        "replica_door": "DoorReplicaCAD-v0",
        "replica_drawer": "DrawerReplicaCAD-v0",

        "replica_faucet": "TurnFaucetReplicaCAD-v0",
        "replica_push": "PushCubeReplicaCAD-v0",
        "replica_lift": "LiftCubeReplicaCAD-v0",

        "pushchair": "FixedChair-v0",
        "movebucket": "MoveBucket-v1",

        "matterport_faucet": "TurnFaucetMatterport-v0",
        "matterport_pickcube": "PickCubeMatterport-v0",
    }

    DOOR_DRAWER_REPLICA_TASKS = [
        "DoorReplicaCAD-v0",
        "DrawerReplicaCAD-v0",
        "TurnFaucetReplicaCAD-v0",
        "PushCubeReplicaCAD-v0",
        "LiftCubeReplicaCAD-v0"]

    ALL_DOOR_DRAWER_TASKS = [
        "OpenCabinetDoor-v1",
        "OpenCabinetDrawer-v1",
        "OpenHackedCabinetDoor-v1",
        "OpenHackedCabinetDrawer-v1",
        "DoorOneHotGoal-v0",
        "DrawerOneHotGoal-v0",
        "DoorReplicaCAD-v0",
        "DrawerReplicaCAD-v0",
    ]

    FAUCET_TASKS = [
        "TurnFaucet-v0",
        "HackedTurnFaucet-v0",
        "TurnFaucetReplicaCAD-v0",
        "TurnFaucetMatterport-v0"
    ]

    CUBE_TASKS = [
        "LiftCube-v0",
        "PickCube-v0",
        "PickCubeMatterport-v0",
        "PushCubeReplicaCAD-v0",
        "LiftCubeReplicaCAD-v0"
    ]

    # ALL_DOOR_DRAWER_TASKS = DOOR_DRAWER_REPLICA_TASKS + DOOR_DRAWER_TASKS

    def __init__(self,
                 task: str,
                 seed: int,
                 obs_type,
                 action_repeat: int = -1,
                 transition_noise_std: float = 0.0,
                 transition_noise_type: str = "white",
                 img_size: tuple[int, int] = (64, 64),
                 image_to_info: bool = False,
                 full_state_to_info: bool = False,
                 control_mode: str = "pose",
                 modify_camera_pose: bool = False,
                 background_name: Optional[str] = None,
                 fixed_seed: bool = False,
                 crop_image_to_square: bool = True,
                 no_torso_rot: bool = False,
                 no_rot: bool = False,
                 fix_target_link: bool = False,
                 success_bonus: float = 0.0,
                 no_termination: bool = True,
                 max_steps: int = 200,
                 env_kwargs: dict = {},
                 camera_cfgs: dict = None):
        import mani_skill2.envs

        super(ManiSkillMBRLEnv, self).__init__(obs_type=obs_type,
                                               image_to_info=image_to_info,
                                               full_state_to_info=full_state_to_info)
        assert task in self.SUPPORTED_TASKS, f"Unsupported task: {task}"
        assert obs_type in self.SUPPORTED_OBS_TYPES, f"Unsupported observation type: {obs_type}"

        if control_mode == "pose":
            control_mode = "pd_ee_delta_pose"
        elif control_mode == "position":
            control_mode = "pd_ee_delta_pos"
        elif control_mode == "joint":
            control_mode = "pd_joint_delta_pos"
        elif control_mode == "joint_vel":
            control_mode = "pd_joint_vel"
        elif control_mode == "joint_target":
            control_mode = "pd_joint_target_delta_pos"
        elif control_mode == "pose_target":
            control_mode = "pd_ee_target_delta_pose"
        elif control_mode == "position_target":
            control_mode = "pd_ee_target_delta_pos"
        else:
            raise AssertionError(f"Unsupported control mode: {control_mode},"
                                 f" supported modes are: pose, position, joint")

        if task in ["drawer",
                    "door",
                    "pushchair",
                    "movebucket",
                    "replica_drawer",
                    "replica_door",
                    "hacked_door",
                    "hacked_drawer",
                    "door_one_hot",
                    "drawer_one_hot"]:
            control_mode = f"base_pd_joint_vel_arm_{control_mode}"

        print("Control Mode: ", control_mode)
        print("Adding", success_bonus, "for success")
        self._success_bonus = success_bonus
        self._no_termination = no_termination

        self._action_repeat = max(1, action_repeat)

        if obs_type == ObsTypes.STATE:
            obs_mode = "state"
        else:
            obs_mode = "rgbd"

        add_segmentation = obs_type in [ObsTypes.SEGMENTATION, ObsTypes.SEGMENTATION_PROPRIOCEPTIVE]

        self._task = self.SUPPORTED_TASKS[task]
        reward_mode = "normalized_dense"
        bg_name = None if background_name is None or background_name.lower() == "none" else background_name
        self._fix_target_link = fix_target_link
        # renderer_kwargs = dict(device="cuda:0") #{}".format(os.environ.get("CUDA_VISIBLE_DEVICES", 0)))
        # print("Renderer kwargs: ", renderer_kwargs)
        if self._task in self.DOOR_DRAWER_REPLICA_TASKS:
            rng = np.random.RandomState(seed=seed)
            # print("Fix Target Link: ", fix_target_link)
            self._base_env = gym.make(id=self._task,
                                      obs_mode=obs_mode,
                                      reward_mode=reward_mode,
                                      render_mode="cameras",
                                      control_mode=control_mode,
                                      rng=rng,
                                      base_path=os.environ["MS2_ASSET_DIR"],
                                      # renderer_kwargs=renderer_kwargs,
                                      #   camera_cfgs=camera_cfgs,
                                      #        fixed_target_link_idx=0 if fix_target_link else None,
                                      **env_kwargs)
        elif self._task in self.DOOR_DRAWER_TASKS:
            print("Fix Target Link: ", fix_target_link)
            self._base_env = gym.make(id=self._task,
                                      obs_mode=obs_mode,
                                      reward_mode=reward_mode,
                                      #    render_mode="cameras",
                                      control_mode=control_mode,
                                      bg_name=bg_name,
                                      # renderer_kwargs=renderer_kwargs,
                                      camera_cfgs=camera_cfgs,
                                      #  fixed_target_link_idx=0 if fix_target_link else None,
                                      **env_kwargs)
        elif "matterport" in self._task.lower():
            print("Using Matterport")
            pose = matterport_camera_poses[self._task]
            env_kwargs.update(matterport_env_kwargs[self._task])
            self._base_env = gym.make(id=self._task,
                                      obs_mode=obs_mode,
                                      reward_mode=reward_mode,
                                      control_mode=control_mode,
                                      bg_name=bg_name,
                                      #   renderer_kwargs=renderer_kwargs,
                                      camera_cfgs=dict(base_camera=dict(width=128,
                                                                        height=128,
                                                                        p=pose.p, q=pose.q)),
                                      **env_kwargs)
        else:
            self._base_env = gym.make(id=self._task,
                                      obs_mode=obs_mode,
                                      reward_mode=reward_mode,
                                      control_mode=control_mode,
                                      bg_name=bg_name,
                                      # renderer_kwargs=renderer_kwargs,
                                      camera_cfgs=camera_cfgs,
                                      **env_kwargs,
                                      )
        if max_steps != 200:
            assert isinstance(self._base_env, gym.wrappers.TimeLimit)
            self._base_env = gym.wrappers.TimeLimit(self._base_env.env, max_episode_steps=max_steps)

        if modify_camera_pose:
            self._modify_camera_pose()
        self._seed = seed
        self._fixed_seed = fixed_seed
        self._img_size = img_size

        if self._task in self.ALL_DOOR_DRAWER_TASKS:
            self._no_torso_rot = no_torso_rot
            self._no_rot = no_rot
            base_act_space = self._base_env.action_space
            if self._no_rot:
                self.action_space = \
                    gym.spaces.Box(low=np.concatenate([base_act_space.low[:2], base_act_space.low[4:]]),
                                   high=np.concatenate([base_act_space.high[:2], base_act_space.high[4:]]),
                                   shape=(base_act_space.shape[0] - 2,), dtype=base_act_space.dtype)
            elif self._no_torso_rot:
                self.action_space = \
                    gym.spaces.Box(low=np.concatenate([base_act_space.low[:3], base_act_space.low[4:]]),
                                   high=np.concatenate([base_act_space.high[:3], base_act_space.high[4:]]),
                                   shape=(base_act_space.shape[0] - 1,), dtype=base_act_space.dtype)
            else:
                self.action_space = base_act_space
        else:
            self._no_rot = self._no_torso_rot = False
            self.action_space = self._base_env.action_space
        self._crop_image_to_square = crop_image_to_square

        self._base_position = None

        self._last_img = None

    @staticmethod
    def _get_ld_space(dim):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=float)

    def _get_goal(self, obs: dict) -> np.ndarray:
        if self._task == "PickCube-v0":
            return obs["extra"]["goal_pos"]
        else:
            raise NotImplementedError(f"Goal observation not implemented for {self._task}")

    def _get_proprioceptive_obs(self, obs: dict) -> np.ndarray:
        if self._task in self.CUBE_TASKS:
            if self._base_position is None:
                self._base_position = obs["agent"]["base_pose"]
            assert np.allclose(self._base_position, obs["agent"]["base_pose"]), "Base pose changed"
            proprio = [obs["agent"]["qpos"], obs["agent"]["qvel"], obs["extra"]["tcp_pose"]]
            # if not self._obs_type == ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL:
            #    proprio.append(obs["extra"]["goal_pos"])
            return np.concatenate(proprio)
        elif self._task in self.FAUCET_TASKS:
            if self._base_position is None:
                self._base_position = obs["agent"]["base_pose"]
            assert np.allclose(self._base_position, obs["agent"]["base_pose"]), "Base pose changed"
            return np.concatenate([obs["agent"]["qpos"],
                                   obs["agent"]["qvel"],
                                   obs["extra"]["tcp_pose"]])
        elif self._task in self.ALL_DOOR_DRAWER_TASKS:
            agent_obs = [obs["agent"]["qpos"],
                         obs["agent"]["qvel"],
                         obs["agent"]["base_pos"],
                         np.array([obs["agent"]["base_orientation"]]),
                         obs["agent"]["base_vel"],
                         np.array([obs["agent"]["base_ang_vel"], ]),
                         obs["extra"]["tcp_pose"],
                         ]
            target_obs = [obs["extra"]["target_joint_axis"],
                          obs["extra"]["target_link_pos"],
                          np.array([obs["extra"]["target_angle_diff"], ])]
            return np.concatenate(agent_obs + target_obs)
        elif self._task in ["PushChair-v1", "MoveBucket-v1", "FixedChair-v0"]:
            agent_obs = [obs["agent"]["qpos"],
                         obs["agent"]["qvel"],
                         obs["agent"]["base_pos"],
                         np.array([obs["agent"]["base_orientation"]]),
                         obs["agent"]["base_vel"],
                         np.array([obs["agent"]["base_ang_vel"], ]),
                         obs["extra"]["left_tcp_pose"],
                         obs["extra"]["right_tcp_pose"]]
            return np.concatenate(agent_obs)
        else:
            raise NotImplementedError(f"Proprioceptive observation not implemented for {self._task}")

    def _get_image(self, obs: dict, gripper: bool, depth: bool, segmentation: bool) -> np.ndarray:
        assert not (depth and segmentation), "Call function twice to get both depth and segmentation"
        key = "depth" if depth else "Segmentation" if segmentation else "rgb"
        if self._task in self.CUBE_TASKS + self.FAUCET_TASKS:
            if gripper:
                img = obs["image"]["hand_camera"][key]
            else:
                img = obs["image"]["base_camera"][key]
        elif self._task in self.ALL_DOOR_DRAWER_TASKS:
            assert not gripper, "Gripper image not implemented for this task"
            if self._crop_image_to_square:
                img = np.ascontiguousarray(obs["image"]["overhead_camera_0"][key][:, 160:320])
            else:
                img = np.ascontiguousarray(obs["image"]["overhead_camera_0"][key])
        elif self._task in ["PushChair-v1", "MoveBucket-v1", "FixedChair-v0"]:
            assert not gripper, "Gripper image not implemented for this task"
            if self._crop_image_to_square:
                img = np.ascontiguousarray(obs["image"]["overhead_camera_0"][key][:, 120:280])
            else:
                img = np.ascontiguousarray(obs["image"]["overhead_camera_0"][key])
        else:
            raise NotImplementedError(f"Image observation not implemented for {self._task}")
        if segmentation:
            target_object_actor_ids = [x.id for x in self._base_env.unwrapped.get_actors() if
                                       x.name not in ['ground', 'goal_site']]

            # get the robot link ids (links are subclass of actors)
            robot_links = self._base_env.unwrapped.agent.robot.get_links()  # e.g., [Actor(name="root", id="1"), Actor(name="root_arm_1_link_1", id="2"), Actor(name="root_arm_1_link_2", id="3"), ...]
            robot_link_ids = np.array([x.id for x in robot_links], dtype=np.int32)

            # obtain segmentations of the target object(s) and the robot
            actor_seg = img[..., 1]
            new_seg = np.ones_like(actor_seg, dtype=np.uint8) * 255
            for _, target_object_actor_id in enumerate(target_object_actor_ids):
                new_seg[np.isin(actor_seg, target_object_actor_id)] = 0
            new_seg[np.isin(actor_seg, robot_link_ids)] = 0
            img = new_seg[..., np.newaxis]
        if depth:
            d_max = 3
            img = (np.clip(img, 0, d_max) * (255 / d_max)).astype(np.uint8)

        img = cv2.resize(img, (self._img_size[1], self._img_size[0]))
        if depth or segmentation:
            img = img[:, :, np.newaxis]
        self._last_img = img
        return img

    def _get_obs(self, obs: Union[dict, np.ndarray]) -> list[np.ndarray]:
        if self._obs_type == ObsTypes.STATE:
            return [obs]
        elif self._obs_type == ObsTypes.PROPRIOCEPTIVE:
            return [self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.IMAGE:
            return [self._get_image(obs, gripper=False, depth=False, segmentation=False)]
        elif self._obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE:
            return [self._get_image(obs, gripper=False, depth=False, segmentation=False),
                    self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.GRIPPER:
            return [self._get_image(obs, gripper=True, depth=False, segmentation=False)]
        elif self._obs_type == ObsTypes.GRIPPER_PROPRIOCEPTIVE:
            return [self._get_image(obs, gripper=True, depth=False, segmentation=False),
                    self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.GRIPPER_GOAL:
            return [self._get_image(obs, gripper=True, depth=False, segmentation=False), self._get_goal(obs)]
        elif self._obs_type == ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL:
            return [self._get_image(obs, gripper=True, depth=False, segmentation=False),
                    self._get_proprioceptive_obs(obs), self._get_goal(obs)]
        elif self._obs_type == ObsTypes.DEPTH:
            return [self._get_image(obs, gripper=False, depth=True, segmentation=False)]
        elif self._obs_type == ObsTypes.DEPTH_PROPRIOCEPTIVE:
            return [self._get_image(obs, gripper=False, depth=True, segmentation=False),
                    self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.SEGMENTATION:
            return [self._get_image(obs, gripper=False, depth=False, segmentation=True)]
        elif self._obs_type == ObsTypes.SEGMENTATION_PROPRIOCEPTIVE:
            return [self._get_image(obs, gripper=False, depth=False, segmentation=True),
                    self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.RGBD:
            c_img = self._get_image(obs, gripper=False, depth=False, segmentation=False)
            d_img = self._get_image(obs, gripper=False, depth=True, segmentation=False)
            return [np.concatenate([c_img, d_img], axis=-1)]
        elif self._obs_type == ObsTypes.RGBD_PROPRIOCEPTIVE:
            c_img = self._get_image(obs, gripper=False, depth=False, segmentation=False)
            d_img = self._get_image(obs, gripper=False, depth=True, segmentation=False)
            return [np.concatenate([c_img, d_img], axis=-1), self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.IMG_DEPTH_PROPRIO:
            c_img = self._get_image(obs, gripper=False, depth=False, segmentation=False)
            d_img = self._get_image(obs, gripper=False, depth=True, segmentation=False)
            c_img = np.ascontiguousarray(c_img)
            d_img = np.ascontiguousarray(d_img)
            return [c_img, d_img, self._get_proprioceptive_obs(obs)]
        elif self._obs_type == ObsTypes.IMG_SEG_PROPRIO:
            return [self._get_image(obs, gripper=False, depth=False, segmentation=False),
                    self._get_image(obs, gripper=False, depth=False, segmentation=True),
                    self._get_proprioceptive_obs(obs)]
        else:
            raise NotImplementedError("Observation type not implemented")

    def step(self, action: np.ndarray) -> tuple[list[np.ndarray], float, bool, bool, dict]:
        rew_accu = 0
        if self._no_rot:
            action = np.concatenate([action[:2], np.zeros(2), action[2:]])
        elif self._no_torso_rot:
            action = np.concatenate([action[:3], np.zeros(1), action[3:]])
        for i in range(self._action_repeat):
            obs, rew, terminated, truncated, info = self._base_env.step(action)
            rew_accu += rew
            terminated = terminated and not self._no_termination

            if info["success"]:
                rew_accu += self._success_bonus
            if truncated or terminated:
                break
        info["success"] = bool(info["success"])
        return self._get_obs(obs=obs), rew_accu, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[list[np.ndarray], dict]:
        obs, info = self._base_env.reset(seed=self._seed)
        if not self._fixed_seed:
            self._seed = None
        return self._get_obs(obs=obs), {"success": False} | info

    def _get_proprio_dim(self):
        if self._task in self.CUBE_TASKS:
            # if self._obs_type == ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL:
            return 25
            # else:
            #   return 28
        elif self._task in self.FAUCET_TASKS:
            return 25
        elif self._task in self.ALL_DOOR_DRAWER_TASKS:
            return 40
        elif self._task in ["PushChair-v1", "MoveBucket-v1", "FixedChair-v0"]:
            return 58

    @property
    def observation_space(self):
        if self._obs_type in [ObsTypes.DEPTH,
                              ObsTypes.SEGMENTATION,
                              ObsTypes.DEPTH_PROPRIOCEPTIVE,
                              ObsTypes.SEGMENTATION_PROPRIOCEPTIVE]:
            channels = 1
        elif self._obs_type in [ObsTypes.RGBD, ObsTypes.RGBD_PROPRIOCEPTIVE]:
            channels = 4
        else:
            channels = 3
        img_obs_space = gym.spaces.Box(low=0, high=255, shape=self._img_size + (channels,), dtype=np.uint8)
        if self._obs_type in [ObsTypes.IMAGE,
                              ObsTypes.GRIPPER,
                              ObsTypes.DEPTH,
                              ObsTypes.SEGMENTATION,
                              ObsTypes.RGBD]:
            return gym.spaces.Tuple([img_obs_space])
        elif self._obs_type in [ObsTypes.IMAGE_PROPRIOCEPTIVE,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE,
                                ObsTypes.DEPTH_PROPRIOCEPTIVE,
                                ObsTypes.SEGMENTATION_PROPRIOCEPTIVE,
                                ObsTypes.RGBD_PROPRIOCEPTIVE]:
            return gym.spaces.Tuple([img_obs_space, self._get_ld_space(dim=self._get_proprio_dim())])
        elif self._obs_type == ObsTypes.STATE:
            return gym.spaces.Tuple([self._base_env.observation_space])
        elif self._obs_type == ObsTypes.PROPRIOCEPTIVE:
            return gym.spaces.Tuple([self._get_ld_space(dim=self._get_proprio_dim())])
        elif self._obs_type == ObsTypes.GRIPPER_GOAL:
            return gym.spaces.Tuple([img_obs_space, self._get_ld_space(dim=3)])
        elif self._obs_type == ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL:
            return gym.spaces.Tuple([img_obs_space,
                                     self._get_ld_space(dim=self._get_proprio_dim()),
                                     self._get_ld_space(dim=3)])
        elif self._obs_type in [ObsTypes.IMG_SEG_PROPRIO, ObsTypes.IMG_DEPTH_PROPRIO]:
            return gym.spaces.Tuple([
                gym.spaces.Box(low=0, high=255, shape=self._img_size + (3,), dtype=np.uint8),
                gym.spaces.Box(low=0, high=255, shape=self._img_size + (1,), dtype=np.uint8),
                self._get_ld_space(dim=self._get_proprio_dim())
            ])

        else:
            raise NotImplementedError("Observation space not implemented")

    @property
    def max_seq_length(self):
        return 200 // self._action_repeat

    def _modify_camera_pose(self):
        if self._task == "PickCube-v0":
            self._base_env.unwrapped._cameras["base_camera"].camera.set_pose(sapien.Pose(p=[0.2, 0, 0.1],
                                                                                         q=[0, 0.135, 0, -0.991]))
        else:
            raise NotImplementedError("Camera pose modification not implemented for this task")

    def render(self):
        return self._base_env.render()
        # assert self._last_img is not None, "No image to render"
        # return self._last_img

    @property
    def normalization_mask(self):
        return None

