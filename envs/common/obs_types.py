from enum import Enum, EnumMeta


class ObsTypes(str, Enum, metaclass=EnumMeta):
    IMAGE = "image"

    # Low Dimension Observations
    STATE = "state"
    POSITION = "position"
    PROPRIOCEPTIVE = "proprioceptive"
    PROPRIOCEPTIVE_POSITION = "proprioceptive_position"

    # Fusion Observations
    IMAGE_STATE = "image_state"
    IMAGE_POSITION = "image_position"
    IMAGE_PROPRIOCEPTIVE = "image_proprioceptive"
    IMAGE_PROPRIOCEPTIVE_POSITION = "image_proprioceptive_position"
    DEPTH = "depth"

    DEPTH_PROPRIOCEPTIVE = "depth_proprioceptive"
    SEGMENTATION_PROPRIOCEPTIVE = "segmentation_proprioceptive"
    RGBD = "rgbd"
    RGBD_PROPRIOCEPTIVE = "rgbd_proprioceptive"
    # Other
    GRIPPER = "gripper"
    GRIPPER_PROPRIOCEPTIVE = "gripper_proprioceptive"
    EXTERNAL_GRIPPER_PROPRIOCEPTIVE = "external_gripper_proprioceptive"

    # Goal Conditioning
    IMAGE_GOAL = "image_goal"
    GRIPPER_GOAL = "gripper_goal"
    IMAGE_PROPRIOCEPTIVE_GOAL = "image_proprioceptive_goal"
    GRIPPER_PROPRIOCEPTIVE_GOAL = "gripper_proprioceptive_goal"
    IMAGE_PROPRIOCEPTIVE_GOAL_CAT = "image_proprioceptive_goal_cat"
    GRIPPER_PROPRIOCEPTIVE_GOAL_CAT = "gripper_proprioceptive_goal_cat"
    EXTERNAL_GRIPPER_PROPRIOCEPTIVE_GOAL = "external_gripper_proprioceptive_goal"
