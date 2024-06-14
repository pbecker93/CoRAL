import torch
import os
import numpy as np

from ssm_rl.common.abstract_evaluator_factory import AbstractEvaluatorFactory
from ssm_rl.util.config_dict import ConfigDict

from ssm_rl.model_learning.evaluation.recon_evaluator import ReconstructionEvaluator
from ssm_rl.model_learning.evaluation.saliency_evaluator import SaliencyEvaluator
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective
from ssm_rl.model_learning.objective_factories import ReconstructionObjectiveFactory

from ssm_rl.policy_learning.common.replay_buffer import ReplayBuffer


def get_model_eval_factory_from_string(eval_name: str) -> AbstractEvaluatorFactory:
    if eval_name == "recon":
        return ReconstructionEvaluatorFactory()
    elif eval_name == "saliency":
        return SaliencyEvaluatorFactory()
    else:
        raise ValueError(f"Unknown evaluator name: {eval_name}.")


def get_eval_buffer(img_preprocessor,
                    env_config,
                    env,
                    obs_type,
                    obs_means,
                    obs_stds,
                    num_seqs: int,
                    base_path: str):

    with torch.inference_mode():
        replay_buffer = ReplayBuffer(add_reward_to_obs=False,
                                     obs_are_images=env.obs_are_images,
                                     dataloader_num_workers=0,
                                     img_preprocessor=img_preprocessor,
                                     max_seqs_in_buffer=-1,
                                     skip_first_n_frames=0)

        env = env_config.env.split("-")[0]
        distr = {"none": "clean", "kinetics": "nat", "disks_medium": "occ"}[env_config.distractor_type]
        folder = os.path.join(base_path, f"rkn_data/evaluation/{env}_{distr}")
        try:
            saved_config = \
                ConfigDict(**dict(np.load(os.path.join(folder, "config.npz"), allow_pickle=True)["env"][None][0]))
            assert saved_config.env == env_config.env
           # assert saved_config.action_repeat == env_config.action_repeat
            assert saved_config.distractor_type == env_config.distractor_type
            assert env_config.img_size == saved_config.img_size
            raw_data = dict(np.load(os.path.join(folder, "data.npz"), allow_pickle=True))
            all_obs, all_actions, all_rewards, all_truncated_flags, all_terminated_flags, all_infos = \
                [], [], [], [], [], []

            for i in range(min(len(raw_data.keys()), num_seqs)):
                sequence = dict(raw_data[str(i)][None][0])
                if obs_type == "img":
                    imgs = sequence["observations"][0]
                    obs = [torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2))))]
                elif obs_type == "img_pro":
                    imgs = sequence["observations"][0]
                    proprio = torch.from_numpy(np.ascontiguousarray(sequence["observations"][1].astype(np.float32)))
                    obs = [torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))), proprio]
                else:
                    raise NotImplementedError("No replay buffer for obs type {}".format(obs_type))
                all_obs.append(obs)
                all_actions.append(torch.from_numpy(np.ascontiguousarray(sequence["actions"].astype(np.float32))))
                all_rewards.append(
                    torch.from_numpy(np.ascontiguousarray(sequence["rewards"][:, None].astype(np.float32))))
                all_truncated_flags.append(torch.from_numpy(np.ascontiguousarray(sequence["truncation_flags"])))
                all_terminated_flags.append(torch.from_numpy(np.ascontiguousarray(sequence["termination_flags"])))
                all_infos.append(
                    {"image": torch.from_numpy(np.ascontiguousarray(np.transpose(sequence["infos"]["image"],
                                                                                 (0, 3, 1, 2))))}
                    if "image" in sequence["infos"].keys() else {}
                )
            if obs_means is not None and obs_stds is not None:
                for i, (mean, std) in enumerate(zip(obs_means, obs_stds)):
                    if mean is not None and std is not None:
                        [(o[i].sub_(mean)).div_(std) for o in all_obs]

            replay_buffer.add_data(obs=all_obs,
                                   actions=all_actions,
                                   rewards=all_rewards,
                                   truncated=all_truncated_flags,
                                   terminated=all_terminated_flags,
                                   infos=all_infos)
            replay_buffer.save_obs_norm_params(obs_means=obs_means, obs_stds=obs_stds)
            replay_buffer.freeze()
            print("Found eval buffer")
            return replay_buffer
        except FileNotFoundError:
            print("No Evaluation Buffer Found - returning None")
            return None


class ReconstructionEvaluatorFactory(AbstractEvaluatorFactory):

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.reconstruct_from_prior = False
        config.save_interval = 20
        config.save_num_seqs = 3

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(model_objective: AbstractModelObjective,
              save_path: str,
              config: ConfigDict,
              fixed_eval_buffer: ReplayBuffer,
              device):
        conv_net = ReconstructionObjectiveFactory.build_conv(
            input_size=model_objective.model.feature_size,
            target_size=(3, 64, 64),
            conv_depth_factor=32,
            activation="ReLU")
        if fixed_eval_buffer is None:
            fixed_eval_loader = None
        else:
            fixed_eval_loader = \
                fixed_eval_buffer.get_data_loader(device=device,
                                                  batch_size=config.save_num_seqs,
                                                  num_batches=1,
                                                  seq_length=32,
                                                  val_dataset=True,
                                                  seq_idx=np.arange(0, config.save_num_seqs),
                                                  start_idx=np.zeros(config.save_num_seqs, dtype=int))

        return ReconstructionEvaluator(model_objective=model_objective,
                                       conv_net=conv_net,
                                       save_path=save_path,
                                       reconstruct_from_prior=config.reconstruct_from_prior,
                                       save_imgs_every=config.save_interval,
                                       save_num_sequences=config.save_num_seqs,
                                       fixed_eval_loader=fixed_eval_loader)

    @staticmethod
    def name() -> str:
        return ReconstructionEvaluator.name()

    def needs_image_in_info(self):
        return True


class SaliencyEvaluatorFactory(AbstractEvaluatorFactory):

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.eval_interval = 2
        config.save_num_imgs = 5
        config.sub_seq_length = 8

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(model_objective: AbstractModelObjective,
              save_path: str,
              config: ConfigDict,
              fixed_eval_buffer: ReplayBuffer,
              device):
        if fixed_eval_buffer is None:
            fixed_eval_loader = None
        else:
            fixed_eval_loader = fixed_eval_buffer.get_data_loader(device=device,
                                                                  batch_size=1,
                                                                  num_batches=config.save_num_imgs,
                                                                  seq_length=config.sub_seq_length,
                                                                  val_dataset=True,
                                                                  seq_idx=np.arange(0, config.save_num_imgs),
                                                                  start_idx=np.zeros(config.save_num_imgs, dtype=int))
        return SaliencyEvaluator(model_objective=model_objective,
                                 save_path=save_path,
                                 obs_idx=0,
                                 save_num_images=config.save_num_imgs,
                                 eval_interval=config.eval_interval,
                                 sub_seq_length=config.sub_seq_length,
                                 fixed_eval_loader=fixed_eval_loader)

    @staticmethod
    def name() -> str:
        return SaliencyEvaluator.name()
