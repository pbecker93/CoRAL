import torch
import time
from typing import Union, Optional

from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy

from ssm_rl.common.bayesian_modules import MAPEstimate
from ssm_rl.util.stack_util import stack_maybe_nested_dicts


class PolicyEnvInterface:

    def __init__(self, policy):
        self._policy = policy

    def _prepare_observation_for_policy(self,
                                        observation: Union[torch.Tensor, list[torch.Tensor]]) -> list[torch.Tensor]:
        if not isinstance(observation, list):
            observation = [observation]
        obs = []
        for o in observation:
            # un batched vector (1) or image (3)
            if len(o.shape) in [1, 3]:
                obs.append(o.unsqueeze(0).to(self._policy.device))
            else:
                obs.append(o.to(self._policy.device))
        return obs

    def _prepare_action_for_policy(self, action: torch.Tensor) -> torch.Tensor:
        return torch.atleast_2d(action).to(self._policy.device)

    def _prepare_obs_valid_for_policy(self, info_dict: dict) -> Optional[list[torch.Tensor]]:
        if "obs_valid" in info_dict.keys():
            return [ov.unsqueeze(0).to(self._policy.device) for ov in info_dict["obs_valid"]]
        else:
            return None


class DataCollector(PolicyEnvInterface):

    def __init__(self,
                 env,
                 policy: AbstractPolicy,
                 sequences_per_collect: int,
                 use_map: bool,
                 max_sequence_length: int = -1,
                 action_noise_std: float = 0.0):

        super().__init__(policy=policy)

        self._env = env
        self._sequences_per_collect = sequences_per_collect
        self._use_map = use_map
        self._max_sequence_length = max_sequence_length
        self._action_noise_std = action_noise_std
        self._global_step = 0

    @property
    def global_step(self):
        return self._global_step

    def collect(self):
        def _collect():
            t0 = time.time()

            with torch.inference_mode():
                all_observations, all_actions, all_rewards, all_terminated, all_truncated, all_infos = \
                    [], [], [], [], [], []
                for i in range(self._sequences_per_collect):
                    observations, actions, rewards, terminated, truncated, infos = self._rollout_policy()
                    all_observations.append(observations)
                    all_actions.append(actions)
                    all_rewards.append(rewards)
                    all_terminated.append(terminated)
                    all_truncated.append(truncated)
                    all_infos.append(infos)
                tf = time.time() - t0
                return all_observations, all_actions, all_rewards, all_terminated, all_truncated, all_infos, tf
        if self._use_map:
            with MAPEstimate(self._policy.model):
                return _collect()
        else:
            return _collect()

    def _rollout_policy(self):

        observations, actions, rewards, terminated_flags, truncated_flags, infos = [], [], [], [], [], []

        obs, info = self._env.reset()
        observations.append(obs)
        actions.append(torch.zeros(self._env.action_space.shape[0]))
        rewards.append(torch.FloatTensor([0.0]))
        terminated_flags.append(torch.BoolTensor([False]))
        truncated_flags.append(torch.BoolTensor([False]))
        infos.append(info)
        action, policy_state = self._policy.get_initial(batch_size=1)
        terminated = truncated = False
        i = 0

        while not (terminated or truncated) and (self._max_sequence_length < 0 or i < self._max_sequence_length):
            obs_for_pol = self._prepare_observation_for_policy(observation=obs)
            action_for_pol = self._prepare_action_for_policy(action=action)
            ov_for_pol = self._prepare_obs_valid_for_policy(info_dict=info)
            action_from_pol, policy_state = self._policy(observation=obs_for_pol,
                                                         prev_action=action_for_pol,
                                                         policy_state=policy_state,
                                                         sample=True,
                                                         obs_valid=ov_for_pol,
                                                         step=self._global_step)
            action = torch.squeeze(action_from_pol, dim=0).cpu()

            if self._action_noise_std > 0:
                action += self._action_noise_std * torch.randn_like(action)
                action.clamp_(min=self._env.action_space.low, max=self._env.action_space.high)

            obs, reward, terminated, truncated, info = self._env.step(action=action)

            self._global_step += 1
            i += 1

            reward = torch.FloatTensor([reward])

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(torch.BoolTensor([terminated]))
            truncated_flags.append(torch.BoolTensor([truncated]))
            infos.append(info)
        observations = [torch.stack([o[i] for o in observations], dim=0) for i in range(len(observations[0]))]

        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        terminated_flags = torch.stack(terminated_flags, dim=0)
        truncated_flags = torch.stack(truncated_flags, dim=0)
        infos = stack_maybe_nested_dicts(infos, dim=0)

        return observations, actions, rewards, terminated_flags, truncated_flags, infos
