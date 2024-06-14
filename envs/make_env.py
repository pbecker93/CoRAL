from typing import Optional

import envs.dmc.factory as dmc_factory
from envs.dmc.dmc_mbrl_env import DMCMBRLEnv
from envs.maniskill.maniskill_mbrl_env import ManiSkillMBRLEnv

from envs.common.obs_types import ObsTypes

import envs.wrapper.factory as wrapper_factory
from envs.wrapper.old_gym_interface_wrapper import OldGymInterfaceWrapper
from envs.wrapper.no_list_wrapper import NoListWrapper
from envs.wrapper.distractors.abstract_source import DistractionSources
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations


def _build_env(domain_name: str,
               task_name: str,
               seed: int,
               obs_type: ObsTypes = ObsTypes.IMAGE,
               action_repeat: int = -1,
               img_size: tuple[int, int] = (64, 64),
               env_kwargs: dict = {},
               transition_noise_type: str = "white",
               transition_noise_std: float = 0.0,
               image_to_info: bool = False,
               full_state_to_info: bool = False):
    if domain_name == "maniskill":
        env = ManiSkillMBRLEnv(task=task_name,
                               seed=seed,
                               obs_type=obs_type,
                               action_repeat=action_repeat,
                               transition_noise_type=transition_noise_type,
                               transition_noise_std=transition_noise_std,
                               img_size=img_size,
                               image_to_info=image_to_info,
                               full_state_to_info=full_state_to_info,
                               control_mode=env_kwargs.pop("control_mode", "pose"),
                               fixed_seed=env_kwargs.get("fixed_seed", False),
                               modify_camera_pose=env_kwargs.get("modify_camera_pose", False),
                               background_name=env_kwargs.get("background_name", None),
                               no_rot=env_kwargs.get("no_rot", False),
                               no_torso_rot=env_kwargs.get("no_torso_rot", False),
                               fix_target_link=env_kwargs.get("fix_target_link", False),
                               crop_image_to_square=env_kwargs.get("crop_image_to_square", True),
                               success_bonus=env_kwargs.get("success_bonus", 0.0),
                               max_steps=env_kwargs.get("max_steps", 200),
                               env_kwargs=env_kwargs.get("env_kwargs", {}),
                               camera_cfgs=env_kwargs.get("camera_cfgs", {}),
                               )
    else:
        env = DMCMBRLEnv(base_env=dmc_factory.build_dmc_base_env(domain_name=domain_name,
                                                                 task_name=task_name,
                                                                 seed=seed,
                                                                 env_kwargs=env_kwargs),
                         seed=seed,
                         obs_type=obs_type,
                         action_repeat=action_repeat,
                         transition_noise_type=transition_noise_type,
                         transition_noise_std=transition_noise_std,
                         img_size=img_size,
                         image_to_info=image_to_info,
                         full_state_to_info=full_state_to_info)
    return env


def make(domain_name: str,
         task_name: str,
         seed: int,
         obs_type: ObsTypes = ObsTypes.IMAGE,
         action_repeat: int = -1,
         img_size: tuple[int, int] = (64, 64),
         transition_noise_type: str = "white",
         transition_noise_std: float = 0.0,
         env_kwargs: dict = {},
         old_gym_return_type: bool = False,
         no_lists: bool = False,

         image_to_info: bool = False,
         full_state_to_info: bool = False,

         distraction_type: Optional[DistractionSources] = None,
         distraction_location: Optional[DistractionLocations] = None,

         ):
    """
    Creates an Environment.
    :param domain_name: Name of the domain (i.e., specifies the robot to use e.g., "cheetah" or "mw_test")
    :param task_name: Name of the task (i.e., specifies the task to perform e.g., "run" (for cheetah) or "reach" (for mw_test))
    :param seed: random seed
    :param obs_type: type of observation space, different types are defined in ObsTypes, different domains support different types
    :param action_repeat: how often to repeat each action, from the viewpoint of the rl algorithm this lowers the action frequency
    :param img_size: size of image observations
    :param distraction_type: type of distraction or occlusion to use (None for no distraction)
    :param distraction_location: where to place the distraction (Foreground for occlusion, background for distraction, both for both, not all types are supported for all grounds)
    :param transition_noise_type: type of noise to add to the transitions (white for white noise, ou for ornstein uhlenbeck noise)
    :param transition_noise_std:
    :param image_to_info: if true, images are dumped to the info dict
    :param full_state_to_info: if true, the full state is dumped to the info dict
    :param env_kwargs: additional kwargs for the environment
    :param old_gym_return_type: if true, the environment uses the old gym return (obs, reward, done, info) instead of the new one (obs, reward, truncated, terminated, info).
                                mostly for backwards compatibility
    :param no_lists: if true, the environment does not return lists for the observations, only possible with obs_types returning a single observation

    :return:
    """

    if distraction_type is not None:
        assert distraction_location is not None, "distraction_ground must be specified"
#        assert "image" in obs_type
        assert wrapper_factory.is_distraction_possible(domain_name, distraction_location), \
            "Invalid domain/ distraction location combination"

    if wrapper_factory.wrapper_will_downscale(obs_type=obs_type,
                                              distraction_type=distraction_type):
        env_img_size = (480, 480)
    else:
        env_img_size = img_size

    env = _build_env(domain_name=domain_name,
                     task_name=task_name,
                     seed=seed,
                     obs_type=obs_type,
                     action_repeat=action_repeat,
                     img_size=env_img_size,
                     env_kwargs=env_kwargs,
                     transition_noise_type=transition_noise_type,
                     transition_noise_std=transition_noise_std,
                     image_to_info=image_to_info,
                     full_state_to_info=full_state_to_info)

    if distraction_type is not None:
        env = wrapper_factory.add_distractors(env=env,
                                              seed=seed,
                                              img_size=img_size,
                                              distraction_type=distraction_type,
                                              distraction_location=distraction_location)

    if no_lists:
        env = NoListWrapper(env=env)

    if old_gym_return_type:
        env = OldGymInterfaceWrapper(env=env)

    return env


def obs_will_be_image(obs_type: ObsTypes) -> list[bool]:
    """
    Given an obs type, returns a list of which observation will be images and which not
    :param obs_type:
    :return:
    """
    if obs_type in [ObsTypes.IMAGE, ObsTypes.GRIPPER]:
        return [True]
    elif obs_type in [ObsTypes.STATE,
                      ObsTypes.POSITION,
                      ObsTypes.DEPTH,
                      ObsTypes.RGBD,
                      ObsTypes.PROPRIOCEPTIVE,
                      ObsTypes.PROPRIOCEPTIVE_POSITION]:
        return [False]
    elif obs_type in [ObsTypes.IMAGE_STATE,
                      ObsTypes.IMAGE_POSITION,
                      ObsTypes.IMAGE_GOAL,
                      ObsTypes.IMAGE_PROPRIOCEPTIVE,
                      ObsTypes.RGBD_PROPRIOCEPTIVE,
                      ObsTypes.DEPTH_PROPRIOCEPTIVE,
                      ObsTypes.GRIPPER_PROPRIOCEPTIVE,
                      ObsTypes.GRIPPER_GOAL,
                      ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION,
                      ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL_CAT,
                      ]:
        return [True, False]
    elif obs_type == ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE:
        return [True, True, False]
    elif obs_type in [ObsTypes.IMAGE_PROPRIOCEPTIVE_GOAL, ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL]:
        return [True, False, False]
    elif obs_type in [ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE_GOAL]:
        return [True, True, False, False]
    else:
        print(obs_type)
        raise AssertionError

