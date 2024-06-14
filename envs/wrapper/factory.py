import numpy as np
from typing import Optional
import yaml
import os

from envs.common.obs_types import ObsTypes

from envs.wrapper.distraction_wrapper import DistractionWrapper
from envs.wrapper.distractors.abstract_source import DistractionSources
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations


def add_distractors(env,
                    seed: int,
                    img_size: tuple[int, int],
                    distraction_type: DistractionSources,
                    distraction_location: str):
    assert np.sum(env.obs_are_images) <= 1, "Only implemented when exactly 1 obs is image"
    return DistractionWrapper(env=env,
                              seed=seed,
                              img_size=img_size,
                              distraction_type=distraction_type,
                              ground=distraction_location,
                              folder=get_distractor_path(distraction_type),
                              img_idx=int(np.argmax(env.obs_are_images)) if any(env.obs_are_images) else 0)


def is_distraction_possible(domain_name: str,
                            distraction_location: str) -> bool:
    invalid_domains = ["ca_empty", "ca_walls", "hurdle_cheetah", "hurdle_walker"]
    invalid_locations = [DistractionLocations.BACKGROUND, DistractionLocations.BOTH]
    return not (domain_name in invalid_domains and distraction_location in invalid_locations)


def get_distractor_path(distraction_type: str) -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__), "../distractor_paths.yml")
    with open(path, "r") as f:
        path_dict = yaml.safe_load(f)
    if distraction_type == DistractionSources.DISKS_MEDIUM:
        return path_dict["pre_rendered_disks"]
    elif distraction_type == DistractionSources.WALLS:
        return path_dict["pre_rendered_walls"]
    elif distraction_type == DistractionSources.DAVIS:
        return path_dict["davis"]
    elif distraction_type == DistractionSources.KINETICS or distraction_type == DistractionSources.KINETICS_GRAY:
        return path_dict["kinetics400"]
    else:
        return None


def wrapper_will_downscale(obs_type: ObsTypes,
                           distraction_type: str) -> bool:
    image_in_obs = obs_type in [ObsTypes.IMAGE,
                                ObsTypes.GRIPPER,
                                ObsTypes.IMAGE_POSITION,
                                ObsTypes.IMAGE_STATE,
                                ObsTypes.IMAGE_PROPRIOCEPTIVE,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE,
                                ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION,
                                ObsTypes.IMAGE_GOAL,
                                ObsTypes.GRIPPER_GOAL,
                                ObsTypes.IMAGE_PROPRIOCEPTIVE_GOAL,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL,
                                ObsTypes.EXTERNAL_GRIPPER_PROPRIOCEPTIVE,
                                ObsTypes.GRIPPER_PROPRIOCEPTIVE_GOAL_CAT]
    downscaling_distraction = DistractionWrapper.will_resize(distraction_type)
    return image_in_obs and downscaling_distraction
