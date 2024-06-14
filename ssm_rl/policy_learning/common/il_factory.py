from ssm_rl.policy_learning.common.wrappers import ObsNormalizationEnvWrapper
from ssm_rl.policy_learning.common.img_preprocessor import *
from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.util.denorm_action_wrapper import DenormActionWrapper


class ImitationLearningFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.train_mode = "joint"

        config.add_subconf("rep_pretrain", ConfigDict())
        config.rep_pretrain.num_epochs = 100
        config.rep_pretrain.steps_per_epoch = 100
        config.rep_pretrain.seq_length = 50
        config.rep_pretrain.batch_size = 50

        config.add_subconf("behavior_train", ConfigDict())
        config.behavior_train.steps_per_epoch = 100
        config.behavior_train.seq_length = 50
        config.behavior_train.batch_size = 50

        config.add_subconf("il_exp", ConfigDict())
        config.il_exp.bypass_mask = None

        config.add_subconf("initial_data_collection", ConfigDict())
        config.initial_data_collection.seq_length = -1
        config.initial_data_collection.num_sequences = 5

        config.add_subconf("img_preprocessing", ConfigDict())
        config.img_preprocessing.type = "none"
        config.img_preprocessing.color_depth_bits = 5
        config.img_preprocessing.add_cb_noise = True
        config.img_preprocessing.pad = 4
        config.img_preprocessing.crop_size = 64

        config.add_subconf("validation", ConfigDict())
        config.validation.num_batches = 1
        config.validation.batch_size = 1000

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build_img_preprocessor(img_sizes: list[tuple[int, int, int]], config: ConfigDict):
        if len(img_sizes) > 1:
            assert all(img_sizes[0] == img_size for img_size in img_sizes), "Images must be of same size"
        if config.type == "none":
            return ImgPreprocessor(input_img_size=(img_sizes[0][1], img_sizes[0][2]))
        elif config.type == "cd_reduction":
            return ColorDepthReductionImgPreprocessor(input_img_size=(img_sizes[0][1], img_sizes[0][2]),
                                                      depth_bits=config.color_depth_bits,
                                                      add_cb_noise=config.add_cb_noise)
        elif config.type == "crop":
            return CropImagePreprocessor(input_img_size=(img_sizes[0][1], img_sizes[0][2]),
                                         crop_size=config.crop_size)
        elif config.type == "shift":
            return ShiftImgPreprocessor(input_img_size=(img_sizes[0][1], img_sizes[0][2]),
                                        pad=config.pad)
        else:
            raise ValueError(f"Unknown image preprocessing type: {config.type}")

    @staticmethod
    def build_env_and_replay_buffer(env,
                                    img_preprocessor: ImgPreprocessor,
                                    rep_learn_buffer_factory,
                                    config: ConfigDict):

        with torch.inference_mode():
            replay_buffer = rep_learn_buffer_factory.build(img_preprocessor=img_preprocessor)

            obs_means, obs_stds = replay_buffer.obs_norm_params
            if obs_means is not None:
                env = ObsNormalizationEnvWrapper(env=env,
                                                 obs_means=obs_means,
                                                 obs_stds=obs_stds)
            act_mean, act_std = replay_buffer.act_norm_params
            if act_mean is not None:
                env = DenormActionWrapper(env=env,
                                          action_mean=act_mean,
                                          action_std=act_std)
            return env, replay_buffer

