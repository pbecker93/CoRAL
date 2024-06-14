import envs
import torch
import cv2
import gym
from gym import spaces
import numpy as np

from ssm_rl.util.torch_env_wrapper.torch_env_wrapper import TorchEnvWrapper
from ssm_rl.util.config_dict import ConfigDict


class CV2ImgResizeWrapper(gym.ObservationWrapper):

    def __init__(self, env, img_size):
        super().__init__(env)
        self.img_size = img_size

    def observation(self, observation):
        resized_observations = []
        for o, is_image in zip(observation, self.env.obs_are_images):
            if is_image:
                resized_observations.append(cv2.resize(o, (self.img_size, self.img_size)))
            else:
                resized_observations.append(o)
        return resized_observations

    @property
    def observation_space(self) -> spaces.Space:
        obs_spaces = []
        for o, is_image in zip(self.env.observation_space, self.env.obs_are_images):
            if is_image:
                obs_spaces.append(spaces.Box(low=0, high=255,
                                             shape=(self.img_size, self.img_size, o.shape[2]), dtype=np.uint8))
            else:
                obs_spaces.append(o)
        return spaces.Tuple(obs_spaces)


class RLEnvFactory:

    OBS_TYPES = {"img": envs.ObsTypes.IMAGE,
                 "pos": envs.ObsTypes.POSITION,
                 "pro": envs.ObsTypes.PROPRIOCEPTIVE,
                 "pro_pos": envs.ObsTypes.PROPRIOCEPTIVE_POSITION,
                 "state": envs.ObsTypes.STATE,
                 "img_pos": envs.ObsTypes.IMAGE_POSITION,
                 "img_pro": envs.ObsTypes.IMAGE_PROPRIOCEPTIVE,
                 "depth": envs.ObsTypes.DEPTH,
                 "depth_pro": envs.ObsTypes.DEPTH_PROPRIOCEPTIVE,
                 "rgbd": envs.ObsTypes.RGBD,
                 "rgbd_pro": envs.ObsTypes.RGBD_PROPRIOCEPTIVE,
                 "img_pro_pos": envs.ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION,
                 "img_state": envs.ObsTypes.IMAGE_STATE,
                 "egp": envs.ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE,
                 "gripper": envs.ObsTypes.GRIPPER,
                 "gripper_pro": envs.ObsTypes.GRIPPER_PROPRIOCEPTIVE,
                 "img_goal": envs.ObsTypes.IMAGE_GOAL,
                 "gripper_goal": envs.ObsTypes.GRIPPER_GOAL,
                 "ip_goal": envs.ObsTypes.IMAGE_PROPRIOCEPTIVE_GOAL,
                 "gp_goal": envs.ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL,
                 "gp_goal_cat": envs.ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL_CAT,
                 "igpg": envs.ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE_GOAL}

    def __init__(self, obs_type):
        self.obs_type = obs_type

    def get_num_obs(self):
        return len(envs.obs_will_be_image(self.OBS_TYPES[self.obs_type]))

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.env = "cheetah_run"
        config.action_repeat = -1  # env default.yml
        config.img_size = 64

        config.env_kwargs = dict()

        config.distractor_type = "none"
        config.distractor_location = "foreground"

        if finalize_adding:
            config.finalize_adding()

        return config

    def build(self,
              seed: int,
              config: ConfigDict,
              image_to_info: bool = False,
              full_state_to_info: bool = False,
              dtype=torch.float32,
              internal_img_size: int = None,):
        import envs

        if "-" in config.env:
            domain_name, task_name = config.env.split("-")
        else:
            env_list = config.env.split("_")
            domain_name = ""
            for s in env_list[:-2]:
                domain_name += s + "_"
            domain_name += env_list[-2]
            task_name = env_list[-1]

        assert self.obs_type in self.OBS_TYPES.keys(), \
            "Invalid observation type, pick one from {}".format(self.OBS_TYPES.keys())

        distactor_types = {"none": None,
                           "dots_constant": envs.DistractionSources.DOTS_CONSTANT,
                           "dots_episodes": envs.DistractionSources.DOTS_EPISODE,
                           "dots_linear": envs.DistractionSources.DOTS_LINEAR,
                           "dots_pendulum": envs.DistractionSources.DOTS_PENDULUM,
                           "dots_quadlink": envs.DistractionSources.DOTS_QUADLINK,
                           "dots_random": envs.DistractionSources.DOTS_RANDOM,
                           "disks_easy": envs.DistractionSources.DISKS_EASY,
                           "disks_medium": envs.DistractionSources.DISKS_MEDIUM,
                           "disks_hard": envs.DistractionSources.DISKS_HARD,
                           "walls": envs.DistractionSources.WALLS,
                           "kinetics": envs.DistractionSources.KINETICS,
                           "kinetics_gray": envs.DistractionSources.KINETICS_GRAY}
        assert config.distractor_type in distactor_types.keys(), \
            "Invalid distractor type, pick one from {}".format(distactor_types.keys())

        distractor_location = {"foreground": envs.DistractionLocations.FOREGROUND,
                               "background": envs.DistractionLocations.BACKGROUND}
        assert config.distractor_location in distractor_location.keys(), \
            "Invalid distractor location, pick one from {}".format(distractor_location.keys())

        env_img_size = config.img_size if internal_img_size is None else internal_img_size
        env = envs.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=seed,
                        obs_type=self.OBS_TYPES[self.obs_type],
                        action_repeat=config.action_repeat,
                        transition_noise_std=0.0,
                        transition_noise_type="white",
                        distraction_type=distactor_types[config.distractor_type],
                        distraction_location=distractor_location[config.distractor_location],
                        img_size=(env_img_size, env_img_size),
                        image_to_info=image_to_info,
                        full_state_to_info=full_state_to_info,
                        env_kwargs=config.env_kwargs.copy(),
                        old_gym_return_type=False)
        if internal_img_size is not None:
            env = CV2ImgResizeWrapper(env, config.img_size)

        env = TorchEnvWrapper(env, dtype)
        return env

