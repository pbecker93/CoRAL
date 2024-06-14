import os.path

from mani_skill2.envs.ms1.open_cabinet_door_drawer import OpenCabinetDoorEnv, OpenCabinetDrawerEnv
from mbrl_envs.maniskill_envs.adapted_envs.fixed_door_drawer import DrawerOneHotGoalEnv, DoorOneHotGoalEnv
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.registration import register_env
from dm_control.utils.rewards import tolerance
from mani_skill2.envs.ms1.push_chair import PushChairEnv

from mani_skill2.utils.sapien_utils import look_at
from transforms3d.euler import euler2quat

from mani_skill2.envs.pick_and_place.pick_cube import (
    PickCubeEnv,
    LiftCubeEnv,
)

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.geometry import angle_distance, transform_points
from mani_skill2.utils.registration import register_env
from scipy.spatial import distance as sdist

scene_configs = [
    ("Baked_sc0_staging_00.glb", [0, 0.2, 0]),
    ("Baked_sc0_staging_01.glb", [-1, 2, 0]),
    ("Baked_sc0_staging_02.glb", [-1.7, 2.5, 0]),  # good example
    ("Baked_sc0_staging_03.glb", [-3.0, 4.0, 0]),
    ("Baked_sc0_staging_04.glb", [-3.0, 1.0, 0]),
    ("Baked_sc0_staging_05.glb", [-3.0, 1.0, 0]),
    ("Baked_sc0_staging_06.glb", [0.75, -1.0, 0]),
    ("Baked_sc0_staging_07.glb", [-3.0, 1.4, 0]),
    ("Baked_sc0_staging_08.glb", [-1.5, 0.8, 0]),
    ("Baked_sc0_staging_09.glb", [-3.0, 1.0, 0]),

    ("Baked_sc0_staging_10.glb", [0, 0, 0]),
    ("Baked_sc0_staging_11.glb", [-1.5, 2.5, 0]),
    ("Baked_sc0_staging_12.glb", [-1.7, 2.5, 0]),  # good example
    ("Baked_sc0_staging_13.glb", [-3.0, 4.4, 0]),
    ("Baked_sc0_staging_14.glb", [-2.5, 1.0, 0]),
    ("Baked_sc0_staging_15.glb", [-3.0, 1.0, 0]),
    ("Baked_sc0_staging_16.glb", [0.75, -1.0, 0]),
    ("Baked_sc0_staging_17.glb", [0.5, -2.5, 0]),
    ("Baked_sc0_staging_18.glb", [-1.5, 1.0, 0]),
    ("Baked_sc0_staging_19.glb", [-3.0, 1.0, 0]),

    ("Baked_sc1_staging_00.glb", [0.3, 0.5, 0]),
    ("Baked_sc1_staging_01.glb", [-1.5, 2.5, 0]),
    ("Baked_sc1_staging_02.glb", [-1.7, 2.5, 0]),  # good example
    ("Baked_sc1_staging_03.glb", [-3.0, 4.0, 0]),
    ("Baked_sc1_staging_04.glb", [-2.5, 1.0, 0]),
    ("Baked_sc1_staging_05.glb", [-3.0, 1.0, 0]),
    ("Baked_sc1_staging_06.glb", [0.75, -1.0, 0]),
    ("Baked_sc1_staging_05.glb", [-3.0, 1.0, 0]),
    ("Baked_sc1_staging_08.glb", [-1.5, 1.0, 0]),
    ("Baked_sc1_staging_09.glb", [-3.0, 1.0, 0]),

    ("Baked_sc1_staging_10.glb", [0.3, 0.5, 0]),
    ("Baked_sc1_staging_11.glb", [-1.5, 2.5, 0]),
    ("Baked_sc1_staging_12.glb", [-1.7, 2.5, 0]),  # good example
    ("Baked_sc1_staging_13.glb", [-3.0, 3.0, 0]),
    ("Baked_sc1_staging_14.glb", [-2.5, 1.0, 0]),
    ("Baked_sc1_staging_15.glb", [-3.0, 1.0, 0]),
    ("Baked_sc1_staging_16.glb", [0.75, -1.0, 0]),
    ("Baked_sc1_staging_17.glb", [-3.0, 1.0, 0]),
    ("Baked_sc1_staging_18.glb", [-1.5, 1.0, 0]),
    ("Baked_sc1_staging_19.glb", [-2.0, 1.0, 0]),

    ("Baked_sc2_staging_00.glb", [0.3, 0.5, 0]),
    ("Baked_sc2_staging_01.glb", [-1.5, 2.5, 0]),
    ("Baked_sc2_staging_02.glb", [-1.7, 2.5, 0]),  # good example
    ("Baked_sc2_staging_03.glb", [-3.0, 4.0, 0]),
    ("Baked_sc2_staging_04.glb", [-2.5, 1.0, 0]),
    ("Baked_sc2_staging_05.glb", [-3.0, 1.0, 0]),
    ("Baked_sc2_staging_06.glb", [0.75, -1.0, 0]),
    ("Baked_sc2_staging_05.glb", [-3.0, 1.0, 0]),
    ("Baked_sc2_staging_08.glb", [-1.5, 1.0, 0]),
    ("Baked_sc2_staging_09.glb", [-3.0, 1.0, 0]),

    ("Baked_sc2_staging_10.glb", [0.3, 0.5, 0]),
    ("Baked_sc2_staging_11.glb", [-1.5, 2.5, 0]),
    ("Baked_sc2_staging_12.glb", [-1.7, 2.5, 0]),  # good example
    ("Baked_sc2_staging_13.glb", [-3.0, 3.0, 0]),
    ("Baked_sc2_staging_14.glb", [-2.5, 1.0, 0]),
    ("Baked_sc2_staging_15.glb", [-3.0, 1.0, 0]),
    ("Baked_sc2_staging_16.glb", [0.6, 0.5, 0]),
    ("Baked_sc2_staging_17.glb", [-3.5, 1.8, 0]),
    ("Baked_sc2_staging_18.glb", [-1.5, 1.0, 0]),
    ("Baked_sc2_staging_19.glb", [-2.0, 1.0, 0]),

    ("Baked_sc3_staging_00.glb", [0.1, 0.5, 0]),
    ("Baked_sc3_staging_01.glb", [-1.5, 2.0, 0]),
    ("Baked_sc3_staging_02.glb", [-0.0, 2.5, 0]),  # good example
    ("Baked_sc3_staging_03.glb", [-1.7, 3.4, 0]),
    ("Baked_sc3_staging_04.glb", [-1.5, 0.8, 0]),
    ("Baked_sc3_staging_05.glb", [-1.8, 0.8, 0]),
    ("Baked_sc3_staging_06.glb", [0.3, 0.8, 0]),
    ("Baked_sc3_staging_07.glb", [-3.0, 1.8, 0]),
    ("Baked_sc3_staging_08.glb", [-2.5, 1.5, 0]),
    ("Baked_sc3_staging_09.glb", [-2.0, 0.7, 0]),

    ("Baked_sc3_staging_10.glb", [0.1, 0.5, 0]),
    ("Baked_sc3_staging_11.glb", [-1.5, 2.5, 0]),
    ("Baked_sc3_staging_12.glb", [-0.3, 2.2, 0]),  # good example
    ("Baked_sc3_staging_13.glb", [-2.5, 4.4, 0]),
    ("Baked_sc3_staging_14.glb", [-2.2, 1.0, 0]),
    ("Baked_sc3_staging_15.glb", [-2.5, 0.8, 0]),
    ("Baked_sc3_staging_16.glb", [0.2, 0.5, 0]),
    ("Baked_sc3_staging_17.glb", [-3.0, 1.8, 0]),
    ("Baked_sc3_staging_18.glb", [-2.5, 1.5, 0]),
    ("Baked_sc3_staging_19.glb", [-3.0, 0.7, 0]),
]


def clip_and_normalize(x, a_min, a_max=None):
    if a_max is None:
        a_max = np.abs(a_min)
        a_min = -a_max
    return (np.clip(x, a_min, a_max) - a_min) / (a_max - a_min)


def _get_replica_cad_env(base_class):

        class _ReplicaCADEnv(base_class):

            def __init__(self, rng, base_path, *args, **kwargs):
                self._rng = rng
                self._base_path = base_path
                super().__init__(*args, **kwargs)

                print(self.model_db.keys())
                print(len(self.model_ids))
                print("######################")
                print("Using model: ", self.model_ids)
                print("#######################")

            def reset(self, seed=None, options=None):
                if options is None:
                    options = dict()
                obs, info = super().reset(seed=seed, options=options)

                if self._reward_mode in ["dense", "normalized_dense"]:
                    info = self.evaluate()
                    self.compute_dense_reward(info=info)
                info["elapsed_steps"] = 0
                return obs, info

            def evaluate(self, **kwargs) -> dict:
                vel_norm = np.linalg.norm(self.target_link.velocity)
                ang_vel_norm = np.linalg.norm(self.target_link.angular_velocity)
                link_qpos = self.link_qpos

                flags = dict(
                    # Old version, aligned with reward:
                    # cabinet_static=vel_norm <= 0.1 and ang_vel_norm <= 1,
                    # New version, not aligned with reward??:
                    cabinet_static=bool(self.check_actor_static(
                        self.target_link, max_v=0.1, max_ang_v=1
                    )),
                    open_enough=bool(link_qpos >= self.target_qpos),
                )

                return dict(
                    success=all(flags.values()),
                    **flags,
                    link_vel_norm=vel_norm,
                    link_ang_vel_norm=ang_vel_norm,
                    link_qpos=link_qpos
                )

            def compute_dense_reward(self, *args, info: dict, **kwargs):
                reward = 0.0

                # -------------------------------------------------------------------------- #
                # The end-effector should be close to the target pose
                # -------------------------------------------------------------------------- #
                handle_pose = self.target_link.pose
                ee_pose = self.agent.hand.pose

                # Position
                ee_coords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
                handle_pcd = transform_points(
                    handle_pose.to_transformation_matrix(), self.target_handle_pcd
                )
                # trimesh.PointCloud(handle_pcd).show()
                disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)
                dist_ee_to_handle = disp_ee_to_handle.reshape(2, -1).min(-1)  # [2]
                reward_ee_to_handle = -dist_ee_to_handle.mean() * 2
                reward += reward_ee_to_handle  # negative

                # Encourage grasping the handle
                ee_center_at_world = ee_coords.mean(0)  # [10, 3]
                ee_center_at_handle = transform_points(
                    handle_pose.inv().to_transformation_matrix(), ee_center_at_world
                )
                # self.ee_center_at_handle = ee_center_at_handle
                dist_ee_center_to_handle = self.target_handle_sdf.signed_distance(
                    ee_center_at_handle
                )
                # print("SDF", dist_ee_center_to_handle)
                dist_ee_center_to_handle = dist_ee_center_to_handle.max()
                reward_ee_center_to_handle = (
                        clip_and_normalize(dist_ee_center_to_handle, -0.01, 4e-3) - 1
                )
                reward += reward_ee_center_to_handle

                # pointer = trimesh.creation.icosphere(radius=0.02, color=(1, 0, 0))
                # trimesh.Scene([self.target_handle_mesh, trimesh.PointCloud(ee_center_at_handle)]).show()

                # Rotation
                target_grasp_poses = self.target_handles_grasp_poses[self.target_link_idx]
                target_grasp_poses = [handle_pose * x for x in target_grasp_poses]
                angles_ee_to_grasp_poses = [
                    angle_distance(ee_pose, x) for x in target_grasp_poses
                ]
                ee_rot_reward = -min(angles_ee_to_grasp_poses) / np.pi * 3
                reward += ee_rot_reward

                # -------------------------------------------------------------------------- #
                # Stage reward
                # -------------------------------------------------------------------------- #
                coeff_qvel = 1.5 # joint velocity
                coeff_qpos = 0.5  # joint position distance

                stage_reward_close_to_handle = 0.5 # Stage 1 gripper close
                finger_close_factor = 0.0  # Stage 1 gripper close
                # print("FCF:", finger_close_factor)
                stage_reward_open_enough = 2.0  # Stage 2 open
                stage_reward_static = 1.0
                # Stage 3 static
                # stage_reward = -5 - (coeff_qvel + coeff_qpos)

                stage_reward = -1.5 - sum([coeff_qvel,
                                           coeff_qpos,
                                           stage_reward_close_to_handle,
                                           stage_reward_open_enough,
                                           stage_reward_static])

                # Legacy version also abstract coeff_qvel + coeff_qpos.

                link_qpos = info["link_qpos"]
                link_qvel = self.link_qvel
                link_vel_norm = info["link_vel_norm"]
                link_ang_vel_norm = info["link_ang_vel_norm"]

                info["link_qvel"] = link_qvel
                info["link_vel_norm"] = link_vel_norm
                info["link_ang_vel_norm"] = link_ang_vel_norm

                ee_close_to_handle = bool(
                    dist_ee_to_handle.max() <= 0.01 and dist_ee_center_to_handle > 0
                )

                inter_finger_distance = np.linalg.norm(ee_coords[0] - ee_coords[1], axis=-1).mean()

                # Stages
                info["ee_close_to_handle"] = ee_close_to_handle
                info["inter_finger_distance"] = inter_finger_distance

                info["static"] = False
                info["reward_static"] = 0.0

                if ee_close_to_handle:
                    stage_reward += stage_reward_close_to_handle

                    # Distance between current and target joint positions
                    # TODO(jigu): the lower bound 0 is problematic? should we use lower bound of joint limits?
                    reward_qpos = (
                            clip_and_normalize(link_qpos, 0, self.target_qpos) * coeff_qpos
                    )
                    reward += reward_qpos

                    reward += finger_close_factor * (1 - clip_and_normalize(inter_finger_distance, 0.0, 0.08))

                    if not info["open_enough"]:
                        # Encourage positive joint velocity to increase joint position
                        q_vel_upper = 0.5
                        reward_qvel = clip_and_normalize(link_qvel, -0.1, q_vel_upper) * coeff_qvel
                        reward += reward_qvel
                    else:
                        # info["open_enough"] = True
                        # Add coeff_qvel for smooth transition of stagess
                        stage_reward += stage_reward_open_enough + coeff_qvel
                        # TODO Clip?
                        reward_static = -(link_vel_norm + link_ang_vel_norm * 0.5)  # 0.6 if success, stage reward 3.5?
                        info["reward_static"] = reward_static
                        reward += reward_static

                        # Legacy version uses static from info, which is incompatible with MPC.
                        # if info["cabinet_static"]:
                        if link_vel_norm <= 0.1 and link_ang_vel_norm <= 1:
                            info["static"] = True
                            stage_reward += stage_reward_static

                # Update info
                info.update(ee_close_to_handle=ee_close_to_handle, stage_reward=stage_reward)

                info["stage_reward"] = stage_reward
                info["reward"] = reward_ee_to_handle

                reward += stage_reward
                return reward

            def _load_actors(self):
                super()._load_actors()

                # -------------------------------------------------------------------------- #
                # Load static scene
                # -------------------------------------------------------------------------- #
                builder = self._scene.create_actor_builder()
                idx = self._rng.randint(0, len(scene_configs))
                path = os.path.join(self._base_path, "stages_uncompressed", scene_configs[idx][0])

                scene_pose = sapien.Pose(p=[0, 0, 0], q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes

                builder.add_visual_from_file(path, scene_pose)
                ##builder.add_collision_from_file(path, scene_pose)
                self.arena = builder.build_static()
                offset = np.array(scene_configs[idx][1])
                self.arena.set_pose(sapien.Pose(offset))

            def _setup_lighting(self):
                shadow = self.enable_shadow
                ambient_light = self._rng.rand(3) * 0.4 + 0.1
                self._scene.set_ambient_light(ambient_light)
                # Only the first of directional lights can have shadow
                self._scene.add_directional_light(
                    [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
                )
                self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
        return _ReplicaCADEnv


door_decorator = (register_env("DoorReplicaCAD-v0", max_episode_steps=200, override=True))
door_decorator(_get_replica_cad_env(OpenCabinetDoorEnv))

drawer_decorator = (register_env("DrawerReplicaCAD-v0", max_episode_steps=200, override=True))
drawer_decorator(_get_replica_cad_env(OpenCabinetDrawerEnv))

door_decorator = (register_env("DoorReplicaCADOneHot-v0", max_episode_steps=200, override=True))
door_decorator(_get_replica_cad_env(DoorOneHotGoalEnv))

drawer_decorator = (register_env("DrawerReplicaCADOneHot-v0", max_episode_steps=200, override=True))
drawer_decorator(_get_replica_cad_env(DrawerOneHotGoalEnv))


QPOS_LOW = np.array(
    [0.0, np.pi * 2 / 8, 0, -np.pi * 5 / 8, 0, np.pi * 7 / 8, np.pi / 4, 0.04, 0.04]
)
QPOS_HIGH = np.array(
    [0.0, np.pi * 1 / 8, 0, -np.pi * 5 / 8, 0, np.pi * 6 / 8, np.pi / 4, 0.04, 0.04]
)

CUBE_HALF_SIZE = 0.02
BASE_POSE = sapien.Pose([-0.615, 0, 0.05])
xyz = np.hstack([0.0, 0.0, CUBE_HALF_SIZE])
quat = np.array([1.0, 0.0, 0.0, 0.0])
OBJ_INIT_POSE = sapien.Pose(xyz, quat)


@register_env("TurnFaucetReplicaCAD-v0", max_episode_steps=200)
class ReplicaTurnFaucet(TurnFaucetEnv):

    MODEL_IDS = [ '5002', '5004', '5005', '5006', '5007',
                  '5018', '5020', '5021', '5023', '5024',
                  '5027', '5028', '5029', '5030', '5033']#,
          #        '5049', '5012', '5016', '5025', '5069']

    CAM_POSE = look_at([0.15, -0.25, 0.35], [0.0, 0.0, 0.1])

    def __init__(self, rng, base_path, *args, **kwargs):
        self._rng = rng
        self._base_path = base_path

        self._base_path = base_path
        assert "camera_cfgs" not in kwargs
        kwargs["camera_cfgs"] = dict(base_camera=dict(width=128,
                                                      height=128,
                                                      p=ReplicaTurnFaucet.CAM_POSE.p,
                                                      q=ReplicaTurnFaucet.CAM_POSE.q
                                                      )
                                    )
        if "model_ids" not in kwargs.keys():
            kwargs["model_ids"] = self.MODEL_IDS

        super().__init__(*args, **kwargs)

        print(self.model_db.keys())
        print(len(self.model_ids))
        print("######################")
        print("Using model: ", self.model_ids)
        print("#######################")

    def _clear(self):
        # Release cached resources
        self._renderer.clear_cached_resources()
        super()._clear()

    def _initialize_agent(self):
        # Set ee to be above the faucet
        self.agent.reset(QPOS_HIGH)
        self.agent_init_pose = BASE_POSE
        self.agent.robot.set_pose(self.agent_init_pose)

    def _initialize_articulations(self):
        q = euler2quat(0, 0, 0)
        p = np.array([0.1, 0.0, 0.0])
        self.faucet.set_pose(sapien.Pose(p, q))

    def _load_actors(self):
        builder = self._scene.create_actor_builder()
        idx = self._rng.randint(0, len(scene_configs))
        path = os.path.join(self._base_path, "stages_uncompressed", scene_configs[idx][0])


        scene_pose = sapien.Pose(p=[0, 0, -0.6], q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes

        builder.add_visual_from_file(path, scene_pose)
        self.arena = builder.build_static()
        offset = np.array(scene_configs[idx][1])
        self.arena.set_pose(sapien.Pose(offset))

    def get_done(self, info, **kwargs):
        # Disable done from task completion
        return False

    def _setup_lighting(self):
        shadow = self.enable_shadow
        ambient_light = self._rng.rand(3) * 0.4 + 0.1
        self._scene.set_ambient_light(ambient_light)
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])


@register_env("PickCubeReplicaCAD-v0", max_episode_steps=200, override=True)
class PickCubeReplicaCAD(PickCubeEnv):

    CAM_POSE = look_at([0.15, 0.25, 0.35], [0.0, 0.0, 0.2])

    def __init__(self, rng, base_path, *args, **kwargs):
        self._rng = rng
        self._base_path = base_path

        kwargs["camera_cfgs"] = dict(base_camera=dict(width=128,
                                                      height=128,
                                                      p=LiftCubeReplicaCAD.CAM_POSE.p,
                                                      q=LiftCubeReplicaCAD.CAM_POSE.q
                                                      )
                                    )

        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict(reconfigure=True)
        else:
            options["reconfigure"] = True
        return super().reset(seed=seed, options=options)

    def _clear(self):
        # Release cached resources
        self._renderer.clear_cached_resources()
        super()._clear()

    def _initialize_task(self):
        pass
        # Fix goal position
        #self.goal_pos = np.array([0.1, 0.0, 0.3])
        #self.goal_site.set_pose(sapien.Pose(self.goal_pos))

    def _initialize_agent(self):
        # Set ee to be near the object
        self.agent.reset(QPOS_LOW)
        self.agent_init_pose = BASE_POSE
        self.agent.robot.set_pose(self.agent_init_pose)

    def _initialize_actors(self):

        x = (self._rng.rand() - 1) * 0.1
        y = self._rng.rand() * 0.1 + 0.05  # '(self._rng.rand(2) - 1) * 0.10
        self.obj_init_pose = sapien.Pose([x, y, OBJ_INIT_POSE.p[2]], OBJ_INIT_POSE.q)
        self.obj.set_pose(self.obj_init_pose)

    def _load_actors(self):
        self._add_ground(render=False)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)


        builder = self._scene.create_actor_builder()
        idx = self._rng.randint(0, len(scene_configs))
        path = os.path.join(self._base_path, "stages_uncompressed", scene_configs[idx][0])

        scene_pose = sapien.Pose(p=[0, 0, -0.6], q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes

        builder.add_visual_from_file(path, scene_pose)
        self.arena = builder.build_static()
        offset = np.array(scene_configs[idx][1])
        self.arena.set_pose(sapien.Pose(offset))

    def get_done(self, info, **kwargs):
        # Disable done from task completion
        return False

    def _setup_lighting(self):
        shadow = self.enable_shadow
        ambient_light = self._rng.rand(3) * 0.4 + 0.1
        self._scene.set_ambient_light(ambient_light)
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])


@register_env("PushCubeReplicaCAD-v0", max_episode_steps=200, override=True)
class PushCubeReplicaCAD(PickCubeReplicaCAD):
    def _initialize_task(self):
        # Fix goal position
        self.goal_pos = np.array([-0.15, 0.15, 0.0])
        self.goal_site.set_pose(sapien.Pose(self.goal_pos))

    def compute_dense_reward(self, info, **kwargs):
        _CUBE_HALF_SIZE = self.cube_half_size[0]
        _GOAL_THRESH = self.goal_thresh

        tcp_to_obj = np.linalg.norm(self.obj.pose.p - self.tcp.pose.p)
        obj_to_goal = np.linalg.norm(self.goal_pos - self.obj.pose.p)
        gripper_dist = np.linalg.norm(
            self.agent.finger1_link.pose.p - self.agent.finger2_link.pose.p
        )

        reaching_reward = tolerance(
            tcp_to_obj,
            bounds=(0, _CUBE_HALF_SIZE),
            margin=np.linalg.norm(self.obj_init_pose.p - self.agent_init_pose.p),
            sigmoid="long_tail",
        )
        reward = reaching_reward

        # Only issue gripping reward if agent is close to object
        if tcp_to_obj < _CUBE_HALF_SIZE:
            # Encourage agent to close gripper
            gripping_reward = tolerance(
                gripper_dist,
                bounds=(0, _CUBE_HALF_SIZE * 2),
                margin=_CUBE_HALF_SIZE,
                sigmoid="linear",
            )
            reward += 0.5 * gripping_reward

        # Only issue pushing reward if object is grasped
        if self.agent.check_grasp(self.obj, max_angle=30):
            # Add placing reward
            pushing_reward = tolerance(
                obj_to_goal,
                bounds=(0, _GOAL_THRESH),
                margin=np.linalg.norm(self.goal_pos - self.obj_init_pose.p),
                sigmoid="linear",
            )
            reward += 5 * pushing_reward
        return reward


@register_env("LiftCubeReplicaCAD-v0", max_episode_steps=200, override=True)
class LiftCubeReplicaCAD(LiftCubeEnv):

    CAM_POSE = look_at([0.15, 0.25, 0.35], [0.0, 0.0, 0.2])

    def __init__(self, rng, base_path, *args, **kwargs):
        self._rng = rng
        self._base_path = base_path
        assert "camera_cfgs" not in kwargs
        kwargs["camera_cfgs"] = dict(base_camera=dict(width=128,
                                                      height=128,
                                                      p=LiftCubeReplicaCAD.CAM_POSE.p,
                                                      q=LiftCubeReplicaCAD.CAM_POSE.q
                                                      )
                                    )
        super().__init__(*args, **kwargs)

    def _clear(self):
        # Release cached resources
        self._renderer.clear_cached_resources()
        super()._clear()

    def _initialize_task(self):
        # Fix goal position
        self.goal_pos = np.array([0.0, 0.0, 0.3])
        self.goal_site.set_pose(sapien.Pose(self.goal_pos))

    def _initialize_agent(self):
        # Set ee to be near the object
        self.agent.reset(QPOS_LOW)
        self.agent_init_pose = BASE_POSE
        self.agent.robot.set_pose(self.agent_init_pose)

    def _initialize_actors(self):
        xy = (self._rng.rand(2) - 1) * 0.10
        self.obj_init_pose = sapien.Pose([xy[0], xy[1], OBJ_INIT_POSE.p[2]], OBJ_INIT_POSE.q)
        self.obj.set_pose(self.obj_init_pose)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict(reconfigure=True)
        else:
            options["reconfigure"] = True
        return super().reset(seed=seed, options=options)

    def _load_actors(self):
     #   super()._load_actors()

        self._add_ground(render=False)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

        builder = self._scene.create_actor_builder()
        idx = self._rng.randint(0, len(scene_configs))
        path = os.path.join(self._base_path, "stages_uncompressed", scene_configs[idx][0])
        #q = self._rng.rand(2) * 2 - 1
        #q = q / np.linalg.norm(q)

        #scene_pose = sapien.Pose(p=[0, 0, -0.6], q=[, 0, q[1], 0])

        scene_pose = sapien.Pose(p=[0, 0, -0.6], q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        #rot = 0.0 #(self._rng.rand() * np.pi / 4) - np.pi / 8
       # scene_pose.set_q(euler2quat(np.pi / 2, 0.0, rot))#1.0))

        builder.add_visual_from_file(path, scene_pose)
        ##builder.add_collision_from_file(path, scene_pose)
        self.arena = builder.build_static()
        offset = np.array(scene_configs[idx][1])
        self.arena.set_pose(sapien.Pose(offset))

        # Load invisible ground
        #self._add_ground(render=False)
        # Load cube
        ## Add goal indicator
        #self.goal_site = self._build_sphere_site(self.goal_thresh)
        # Load arena
        #builder = self._scene.create_actor_builder()
        #self.arena = load_Matterport(builder)

    def get_done(self, info, **kwargs):
        # Disable done from task completion
        return False

    def _setup_lighting(self):
        shadow = self.enable_shadow
        ambient_light = self._rng.rand(3) * 0.4 + 0.1
        self._scene.set_ambient_light(ambient_light)
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

@register_env("PushChairReplicaCAD-v0")
class PushChairReplicaCAD(PushChairEnv):

    def __init__(self, rng, base_path, *args, **kwargs):
        self._rng = rng
        self._base_path = base_path
        super().__init__(*args, **kwargs)

        print(self.model_db.keys())
        print(len(self.model_ids))
        print("######################")
        print("Using model: ", self.model_ids)
        print("#######################")

    def _load_actors(self):
        super()._load_actors()

        # -------------------------------------------------------------------------- #
        # Load static scene
        # -------------------------------------------------------------------------- #
        builder = self._scene.create_actor_builder()
        idx = self._rng.randint(0, len(scene_configs))
        path = os.path.join(self._base_path, "stages_uncompressed", scene_configs[idx][0])

        scene_pose = sapien.Pose(p=[0, 0, 0], q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes

        builder.add_visual_from_file(path, scene_pose)
        ##builder.add_collision_from_file(path, scene_pose)
        self.arena = builder.build_static()
        offset = np.array(scene_configs[idx][1])
        self.arena.set_pose(sapien.Pose(offset))

    def _setup_lighting(self):
        shadow = self.enable_shadow
        ambient_light = self._rng.rand(3) * 0.4 + 0.1
        self._scene.set_ambient_light(ambient_light)
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
