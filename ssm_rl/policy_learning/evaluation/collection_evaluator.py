import numpy as np
import torch
import os
import pathlib

from ssm_rl.common.bayesian_modules import MAPEstimate
from ssm_rl.policy_learning.common.data_collector import PolicyEnvInterface
from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy

nn = torch.nn


class CollectionEvaluator(PolicyEnvInterface, nn.Module):

    def __init__(self,
                 env,
                 policy: AbstractPolicy,
                 save_path: str,
                 num_sequences: int,
                 collect_at_mean: bool,
                 collect_at_map: bool,
                 regenerate_obs: bool,
                 collection_obs_type: str,
                 save_original_obs: bool,
                 interval: int):
        nn.Module.__init__(self)
        PolicyEnvInterface.__init__(self, policy=policy)

        self._env = env
        self._num_sequences = num_sequences
        self._collect_at_mean = collect_at_mean
        self._collect_at_map = collect_at_map
        self._interval = interval
        self._regenerate_obs_from_state = regenerate_obs
        self._collection_obs_type = collection_obs_type
        self._save_original_obs = save_original_obs
        self._save_path = os.path.join(save_path, "data_collection")

        if save_path is not None:
            pathlib.Path(self._save_path).mkdir(parents=True, exist_ok=True)

    def evaluate(self,
                 iteration: int):
        def _evaluate():
            with torch.inference_mode():
                if iteration % self._interval == 0 and iteration > 0:
                    all_sequences = {}
                    avg_return = 0
                    for i in range(self._num_sequences):
                        sequence, episode_return = self._rollout_policy()
                        all_sequences[str(i)] = sequence
                        avg_return += episode_return / self._num_sequences

                    np.savez_compressed(os.path.join(self._save_path, "data_collection_{:05d}.npz".format(iteration)),
                                        **all_sequences)
                    return {"avg_return": avg_return}
                else:
                    return {}
        if self._collect_at_map:
            with MAPEstimate(self._policy.model):
                self._policy.model.eval()
                ret_value = _evaluate()
                self._policy.model.train()
                return ret_value
        else:
            return _evaluate()

    def _rollout_policy(self):

        obs, info = self._env.reset(eval_mode=True)
        all_actions = [np.zeros((self._env.action_space.shape[0], ), dtype=np.float32)]
        all_rewards = [np.zeros((1, ), dtype=np.float32)]
        all_termination_flags = [np.zeros((1, ), dtype=np.bool)]
        all_truncation_flags = [np.zeros((1, ), dtype=np.bool)]
        if not self._regenerate_obs_from_state:
            all_obs = [o.cpu().numpy() for o in obs]
            all_infos = [{k: v.cpu().numpy() for k, v in info.items()}]
        else:
            assert self._env.obs_type == "state"
            if self._save_original_obs:
                new_obs, original_obs = self._env.get_current_obs(
                    obs_type={"img_pro": "image_proprioceptive"}.get(self._collection_obs_type,
                                                                     self._collection_obs_type),
                    return_original_image=True)
                obs_info = {"image": original_obs}
            else:
                new_obs = self._env.get_current_obs(
                    obs_type={"img_pro": "image_proprioceptive"}.get(self._collection_obs_type,
                                                                     self._collection_obs_type))
                obs_info = {}
            all_obs = [new_obs]
            original_state = [o.cpu().numpy() for o in obs]
            all_infos = [{k: v.cpu().numpy() for k, v in info.items()} | {"original_state": original_state} | obs_info]

        action, policy_state = self._policy.get_initial(batch_size=1)

        terminated = truncated = False
        episode_return = 0

        while not (terminated or truncated):

            obs_for_pol = self._prepare_observation_for_policy(observation=obs)
            action_for_pol = self._prepare_action_for_policy(action=action)
            ov_for_pol = self._prepare_obs_valid_for_policy(info_dict=info)
            action_from_pol, policy_state = self._policy(observation=obs_for_pol,
                                                         prev_action=action_for_pol,
                                                         policy_state=policy_state,
                                                         sample=not self._collect_at_mean,
                                                         obs_valid=ov_for_pol)
            action = torch.squeeze(action_from_pol, dim=0).cpu()
            obs, reward, terminated, truncated, info = self._env.step(action=action)
            all_actions.append(action.cpu().numpy())
            all_rewards.append(reward.cpu().numpy())
            all_termination_flags.append(terminated.cpu().numpy())
            all_truncation_flags.append(truncated.cpu().numpy())
            if not self._regenerate_obs_from_state:
                all_obs.append([o.cpu().numpy() for o in obs])
                all_infos.append({k: v.cpu().numpy() for k, v in info.items()})
            else:
                assert self._env.obs_type == "state"
                if self._save_original_obs:
                    new_obs, original_obs = self._env.get_current_obs(
                        obs_type={"img_pro": "image_proprioceptive"}.get(self._collection_obs_type,
                                                                         self._collection_obs_type),
                        return_original_image=True)
                    obs_info = {"image": original_obs}
                else:
                    new_obs = self._env.get_current_obs(
                        obs_type={"img_pro": "image_proprioceptive"}.get(self._collection_obs_type,
                                                                         self._collection_obs_type))
                    obs_info = {}
                all_obs.append(new_obs)
                original_state = [o.cpu().numpy() for o in obs]
                all_infos.append({k: v.cpu().numpy() for k, v in info.items()} |
                                 {"original_state": original_state} | obs_info)

            episode_return += reward.item()

        all_obs = [np.stack([o[i] for o in all_obs], axis=0) for i in range(len(all_obs[0]))]
        all_actions = np.stack(all_actions, axis=0)
        all_rewards = np.stack(all_rewards, axis=0)
        all_termination_flags = np.stack(all_termination_flags, axis=0)
        all_truncation_flags = np.stack(all_truncation_flags, axis=0)
        all_infos = {k: np.stack([info[k] for info in all_infos], axis=0) for k in all_infos[0].keys()}
        sequence = {
            "observations": all_obs,
            "actions": all_actions,
            "rewards": all_rewards,
            "termination_flags": all_termination_flags,
            "truncation_flags": all_truncation_flags,
            "infos": all_infos
        }
        return sequence, episode_return

    @staticmethod
    def name() -> str:
        return "collection_eval"
