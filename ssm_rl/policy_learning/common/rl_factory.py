import time
import torch

from ssm_rl.policy_learning.common.data_collector import DataCollector
from ssm_rl.policy_learning.common.wrappers import ObsNormalizationEnvWrapper
from ssm_rl.policy_learning.common.replay_buffer import ReplayBuffer
from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy
from ssm_rl.policy_learning.common.img_preprocessor import *

from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.util.stack_util import stack_maybe_nested_dicts

import warnings


class MBRLFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.add_subconf("initial_data_collection", ConfigDict())
        config.initial_data_collection.seq_length = -1
        config.initial_data_collection.num_sequences = 5

        config.add_subconf("data_collection", ConfigDict())
        config.data_collection.seq_length = -1
        config.data_collection.num_sequences = 1
        config.data_collection.action_noise_std = 0.3

        config.add_subconf("rl_exp", ConfigDict())
        config.rl_exp.bypass_mask = "none"
        config.rl_exp.normalize_obs = True
        config.rl_exp.model_updt_steps = 100
        config.rl_exp.model_updt_seq_length = 50
        config.rl_exp.model_updt_batch_size = 50
        config.rl_exp.replay_buffer_size = -1

        config.add_subconf("img_preprocessing", ConfigDict())
        config.img_preprocessing.type = "none"
        config.img_preprocessing.color_depth_bits = 5
        config.img_preprocessing.add_cb_noise = True
        config.img_preprocessing.pad = 4
        config.img_preprocessing.crop_size = 64

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
                                    config: ConfigDict):

        with torch.inference_mode():
            if config.rl_exp.replay_buffer_size > 0:
                max_seqs_in_buffer = config.rl_exp.replay_buffer_size // env.max_seq_length
            else:
                max_seqs_in_buffer = -1

            replay_buffer = ReplayBuffer(add_reward_to_obs=False,
                                         obs_are_images=env.obs_are_images,
                                         dataloader_num_workers=0,
                                         img_preprocessor=img_preprocessor,
                                         max_seqs_in_buffer=max_seqs_in_buffer,
                                         skip_first_n_frames=0)

            initial_obs, initial_acts, initial_rewards, initial_truncated, initial_terminated, initial_infos = \
                MBRLFactory.collect_initial_data(env=env,
                                                 num_sequences=config.initial_data_collection.num_sequences,
                                                 sequence_length=config.initial_data_collection.seq_length)
            # can be a list than its not equal to
            if isinstance(config.rl_exp.normalize_obs, bool):
                if config.rl_exp.normalize_obs:
                    normalize_obs = list(not is_image for is_image in env.obs_are_images)
                else:
                    normalize_obs = [False] * len(env.obs_are_images)
            elif isinstance(config.rl_exp.normalize_obs, list):
                normalize_obs = list((not is_image) and norm for is_image, norm in
                                     zip(env.obs_are_images, config.rl_exp.normalize_obs))
            else:
                raise AssertionError("Invalid value for config.rl_exp.normalize_obs")
            if any(normalize_obs):
                obs_means, obs_stds = MBRLFactory._inplace_normalize(inputs=initial_obs,
                                                                     normalize=normalize_obs)
                env = ObsNormalizationEnvWrapper(env=env,
                                                 obs_means=obs_means,
                                                 obs_stds=obs_stds)
                replay_buffer.save_obs_norm_params(obs_means=obs_means,
                                                   obs_stds=obs_stds)
            replay_buffer.add_data(obs=initial_obs,
                                   actions=initial_acts,
                                   rewards=initial_rewards,
                                   truncated=initial_truncated,
                                   terminated=initial_terminated,
                                   infos=initial_infos)
            return env, replay_buffer

    @staticmethod
    def build_data_collector(env,
                             policy: AbstractPolicy,
                             config: ConfigDict):
        return DataCollector(env=env,
                             policy=policy,
                             sequences_per_collect=config.data_collection.num_sequences,
                             max_sequence_length=config.data_collection.seq_length,
                             action_noise_std=config.data_collection.action_noise_std,
                             use_map=True)

    @staticmethod
    def _inplace_normalize(inputs: list[list[torch.Tensor]],
                           normalize: list[bool]):
        means, stds = [], []
        for i in range(len(inputs[0])):
            if not normalize[i]:
                means.append(None)
                stds.append(None)
            else:
                s, m = torch.std_mean(torch.cat([obs[i] for obs in inputs], dim=0), dim=0)
                if (s < 1e-2).any():
                    warnings.warn("Clipping std from below")
                    s = s.where(s > 1e-2, torch.ones_like(s))
                means.append(m)
                stds.append(s)
                [(inpt[i].sub_(m)).div_(s) for inpt in inputs]
        return means, stds

    @staticmethod
    def sample_action(action_space):
        eps = torch.rand(action_space.shape[0])
        return (action_space.high - action_space.low) * eps + action_space.low

    @staticmethod
    def _rollout(env,
                 sequence_length: int):
        observations, actions, rewards, terminated_flags, truncated_flags, infos = [], [], [], [], [], []
        obs, info = env.reset()
        observations.append(obs)
        actions.append(torch.zeros(env.action_space.shape[0]))
        rewards.append(torch.FloatTensor([0.0]))
        terminated_flags.append(torch.BoolTensor([False]))
        truncated_flags.append(torch.BoolTensor([False]))
        infos.append(info)
        terminated = truncated = False
        i = 0
        while not (terminated or truncated) and (sequence_length < 0 or i < sequence_length):
            action = MBRLFactory.sample_action(action_space=env.action_space)
            obs, reward, terminated, truncated, info = env.step(action=action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            infos.append(info)
            i += 1
        return observations, actions, rewards, terminated_flags, truncated_flags, infos

    @staticmethod
    def collect_initial_data(env,
                             num_sequences: int,
                             sequence_length: int = -1):
        t0 = time.time()
        with torch.inference_mode():
            all_observations, all_actions, all_rewards, all_terminated, all_truncated, all_infos = \
                [], [], [], [], [], []

            for i in range(num_sequences):
                observations, actions, rewards, terminated, truncated, infos = \
                    MBRLFactory._rollout(env=env, sequence_length=sequence_length)
                no = len(observations[0])
                all_observations.append([torch.stack([o[i] for o in observations], dim=0) for i in range(no)])
                all_actions.append(torch.stack(actions, dim=0))
                all_rewards.append(torch.stack(rewards, dim=0))
                all_terminated.append(torch.stack(terminated, dim=0))
                all_truncated.append(torch.stack(truncated, dim=0))
                all_infos.append(stack_maybe_nested_dicts(infos, dim=0))
            print("Random collection took", time.time() - t0)
            return all_observations, all_actions, all_rewards, all_terminated, all_truncated, all_infos
