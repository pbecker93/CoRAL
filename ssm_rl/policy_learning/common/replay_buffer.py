import os
from typing import Optional, Iterable
import torch
import random
import numpy as np
from ssm_rl.policy_learning.common.img_preprocessor import AbstractImgPreprocessor
data = torch.utils.data


class ReplayBuffer:

    def __init__(self,
                 add_reward_to_obs: bool,
                 obs_are_images: list[bool],
                 img_preprocessor: AbstractImgPreprocessor,
                 dataloader_num_workers: int,
                 max_seqs_in_buffer: int = -1,
                 skip_first_n_frames: int = 0):

        self._add_reward_to_obs = add_reward_to_obs
        self._all_data = []

        self._write_idx = 0
        self._max_seqs_in_buffer = max_seqs_in_buffer

        self._obs_means = None
        self._obs_stds = None

        self._action_mean = None
        self._action_std = None

        self._obs_are_images = obs_are_images

        self._img_preprocessor = img_preprocessor
        self._frozen = False

        self._dataloader_num_workers = dataloader_num_workers

        self._skip_first_n_frames = skip_first_n_frames
        self._relevant_info_keys = ["obs_valid", "loss_mask", "image", "state"]

    def save_obs_norm_params(self,
                             obs_means: list[torch.Tensor],
                             obs_stds: list[torch.Tensor]):
        self._obs_means = obs_means
        self._obs_stds = obs_stds

    @property
    def obs_norm_params(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self._obs_means, self._obs_stds

    def save_act_norm_params(self,
                             action_mean: torch.Tensor,
                             action_std: torch.Tensor):
        self._action_mean = action_mean
        self._action_std = action_std

    @property
    def act_norm_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._action_mean, self._action_std


    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self):
        self._frozen = True

    def _get_seq_dict(self,
                      obs_seq: list[torch.Tensor],
                      act_seq: torch.Tensor,
                      reward_seq: torch.Tensor,
                      terminated_seq: torch.Tensor,
                      truncated_seq: torch.Tensor,
                      info_seq: dict[str, torch.Tensor]):
        return {"obs": [o.to(device="cpu") for o in obs_seq],
                "actions": act_seq.to(device="cpu"),
                "rewards": reward_seq.to(device="cpu"),
                "terminated": terminated_seq.to(device="cpu"),
                "truncated": truncated_seq.to(device="cpu"),
                "infos": {k: v.to(device="cpu") if isinstance(v, torch.Tensor) else v for k, v in info_seq.items()}}

    def add_data(self,
                 obs: list[list[torch.Tensor]],
                 actions: list[torch.Tensor],
                 rewards: list[torch.Tensor],
                 terminated: list[bool],
                 truncated: list[bool],
                 infos: list[dict]):
        assert not self.is_frozen
        for obs_seq, act_seq, reward_seq, ter_seq, tru_seq, info_seq in zip(obs,
                                                                            actions,
                                                                            rewards,
                                                                            terminated,
                                                                            truncated,
                                                                            infos):
            if self._max_seqs_in_buffer > 0 and self._all_data == self._max_seqs_in_buffer:
                self._all_data[self._write_idx] = \
                    self._get_seq_dict(obs_seq=obs_seq,
                                       act_seq=act_seq,
                                       reward_seq=reward_seq,
                                       terminated_seq=ter_seq,
                                       truncated_seq=tru_seq,
                                       info_seq=info_seq)
                self._write_idx = (self._write_idx + 1) % self._max_seqs_in_buffer
            else:
                self._all_data.append(self._get_seq_dict(obs_seq=obs_seq,
                                      act_seq=act_seq,
                                      reward_seq=reward_seq,
                                      terminated_seq=ter_seq,
                                      truncated_seq=tru_seq,
                                      info_seq=info_seq))

    @property
    def has_loss_masks(self) -> bool:
        return self._has_loss_masks

    @property
    def has_obs_valid(self) -> bool:
        return "obs_valid" in self._all_data[0]["infos"].keys()

    def get_data_loader(self,
                        device: torch.device,
                        batch_size: int,
                        num_batches: int,
                        seq_length: int,
                        val_dataset: bool,
                        seq_idx=None,
                        start_idx=None) -> data.DataLoader:
        obs, acts, rewards, terminated, truncated, info, shuffle_train_set =\
            self._sample_seqs(batch_size=batch_size,
                              num_batches=num_batches,
                              seq_length=seq_length,
                              seq_idx=seq_idx,
                              start_idx=start_idx)

        obs = [torch.stack([o[i] for o in obs], dim=0) for i in range(len(obs[0]))]
        actions = torch.stack(acts, dim=0)
        rewards = torch.stack(rewards, dim=0)
        terminated = torch.stack(terminated, dim=0)
        truncated = torch.stack(truncated, dim=0)
        info_for_dl = {}
        for k in info[0].keys():
            if isinstance(info[0][k], list):
                info_for_dl[k] = [torch.stack([i[k][j] for i in info], dim=0) for j in range(len(info[0][k]))]
            else:
                info_for_dl[k] = torch.stack([i[k] for i in info], dim=0)

        return self._build_dataloader(batch_size=batch_size,
                                      device=device,
                                      obs=obs,
                                      prev_actions=actions,
                                      rewards=rewards,
                                      terminated=terminated,
                                      truncated=truncated,
                                      infos=info_for_dl,
                                      val_dataset=val_dataset)

    def _sample_seqs(self,
                     batch_size: int,
                     num_batches: int,
                     seq_length: int,
                     seq_idx=None,
                     start_idx=None):

        num_samples = batch_size * num_batches
        if seq_idx is None:
            seq_idx = np.random.randint(low=0, high=len(self._all_data), size=num_samples)
        else:
            assert len(seq_idx) == num_samples
        if start_idx is not None:
            assert len(start_idx) == num_samples
        obs, acts, rewards, terminated, truncated, infos = [], [], [], [], [], []
        for i, si in enumerate(seq_idx):
            if start_idx is None:
                start = np.random.randint(low=self._skip_first_n_frames,
                                          high=self._all_data[si]["actions"].shape[0] - seq_length)
            else:
                start = start_idx[i]
                assert self._skip_first_n_frames <= start <= self._all_data[si]["actions"].shape[0] - seq_length
            time_slice = slice(start, start + seq_length)
            obs.append([o[time_slice] for o in self._all_data[si]["obs"]])
            acts.append(self._all_data[si]["actions"][time_slice])
            rewards.append(self._all_data[si]["rewards"][time_slice])
            terminated.append(self._all_data[si]["terminated"][time_slice])
            truncated.append(self._all_data[si]["truncated"][time_slice])
            info = {}
            for k, v in self._all_data[si]["infos"].items():
                if k in self._relevant_info_keys:
                    if isinstance(v, list):
                        info[k] = [vv[time_slice] for vv in v]
                    else:
                        info[k] = v[time_slice]
            infos.append(info)

        return obs, acts, rewards, terminated, truncated, infos, False

    def _build_dataloader(self,
                          batch_size: int,
                          device: torch.device,
                          obs: list[torch.Tensor],
                          prev_actions: torch.Tensor,
                          rewards: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: dict,
                          val_dataset: bool = False):

        idx_dict = {}
        data_tensors = []

        for i, o in enumerate(obs):
            idx_dict["obs_{}".format(i)] = len(data_tensors)
            data_tensors.append(o)

        idx_dict["actions"] = len(data_tensors)
        data_tensors.append(prev_actions)

        idx_dict["rewards"] = len(data_tensors)
        data_tensors.append(rewards)

        idx_dict["terminated"] = len(data_tensors)
        data_tensors.append(terminated)

        idx_dict["truncated"] = len(data_tensors)
        data_tensors.append(truncated)

        for k, v in infos.items():
            if isinstance(v, list):
                for i, vv in enumerate(v):
                    idx_dict["infos_{}_{}/{}".format(k, i, len(v))] = len(data_tensors)
                    data_tensors.append(vv)
            else:
                idx_dict["infos_{}".format(k)] = len(data_tensors)
                data_tensors.append(v)

        data_set = data.TensorDataset(*data_tensors)

        def collate_fn(data_batch):
            return self._collate_fn(data_batch=data_batch,
                                    num_obs=len(obs),
                                    idx_dict=idx_dict,
                                    device=device,
                                    val_dataset=val_dataset)

        def seed_worker(worker_id):
            worker_seed = (worker_id + torch.initial_seed()) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return data.DataLoader(data_set,
                               shuffle=False,
                               batch_size=batch_size,
                               num_workers=self._dataloader_num_workers,
                               worker_init_fn=seed_worker,
                               collate_fn=collate_fn)

    def _collate_fn(self,
                    data_batch: list[torch.Tensor],
                    num_obs: int,
                    idx_dict: dict,
                    val_dataset: bool,
                    device: torch.device):
        data_batch = data._utils.collate.default_collate(data_batch)

        if "cuda" in device.type:
            data_batch = data._utils.pin_memory.pin_memory(data_batch)

        obs_batch = []
        for i in range(num_obs):
            obs = data_batch[idx_dict["obs_{}".format(i)]].to(device, non_blocking=True)
            obs_batch.append(obs)

        prev_action_batch = data_batch[idx_dict["actions"]].to(device, non_blocking=True)
        rewards = data_batch[idx_dict["rewards"]].to(device, non_blocking=True)
        terminated = data_batch[idx_dict["terminated"]].to(device, non_blocking=True)
        truncated = data_batch[idx_dict["truncated"]].to(device, non_blocking=True)

        infos = {}
        for k, v in idx_dict.items():
            if k.startswith("infos_"):
                k_list = k.split("_")
                try:
                    list_name = "_".join(k_list[1:-1])
                    list_idx, list_len = [int(s) for s in k_list[-1].split("/")]
                    if list_name not in infos.keys():
                        infos[list_name] = [None] * list_len
                    infos[list_name][list_idx] = data_batch[v].to(device, non_blocking=True)
                except ValueError as _:
                    k = "_".join(k_list[1:])
                    infos[k] = data_batch[v].to(device, non_blocking=True)

        for i, obs in enumerate(obs_batch):
            if self._obs_are_images[i]:
                obs_batch[i] = self._img_preprocessor(obs, eval=val_dataset)
        if "image" in infos.keys():
            infos["image"] = self._img_preprocessor(infos["image"], eval=val_dataset)
        return obs_batch, prev_action_batch, rewards, terminated, truncated, infos # obs_valid, loss_masks

    def save_to_disk(self, path: str):
        save_dict = {"seq_{}".format(i): v for i, v in enumerate(self._all_data)}
        if self._obs_means is not None:
            for i in range(len(self._obs_means)):
                if self._obs_means[i] is not None:
                    save_dict["obs_mean_{}".format(i)] = self._obs_means[i]
                    save_dict["obs_std_{}".format(i)] = self._obs_stds[i]
        print("Saving replay buffer to {}".format(path))
        torch.save(save_dict, os.path.join(path, "replay_buffer.pt"))

    def dump_to_disk(self,
                     path: str,
                     use_keys: Optional[str] = None):
        all_obs, all_actions, all_rewards, _ = self._collect_values(use_keys=use_keys)
        assert isinstance(all_obs[0], list)
        save_dict = {
            "actions": all_actions,
            "rewards": all_rewards,
            "infos": self._collect_infos(use_keys=use_keys)}
        for i in range(len(all_obs[0])):
            save_dict["obs_{}".format(i)] = [obs[i] for obs in all_obs]
            if self._obs_means is not None:
                save_dict["obs_mean_{}".format(i)] = self._obs_means[i]
                save_dict["obs_std_{}".format(i)] = self._obs_stds[i]

        torch.save(save_dict, os.path.join(path, "raw_replay_buffer_dump.pt"))

    def load(self,
             path: str):
        assert not self.is_frozen
        data = torch.load(path)
        normalization_params = {}
        for k, v in data.items():
            if k.startswith("seq_"):
                self.add_data(**{kk: [vv] for kk, vv in v.items()})
            else:
                normalization_params[k] = v
        if len(normalization_params) > 0:
            obs_means, obs_stds = [], []
            for i in range(len(normalization_params) // 2):
                obs_means.append(normalization_params["obs_mean_{}".format(i)])
                obs_stds.append(normalization_params["obs_std_{}".format(i)])
            return obs_means, obs_stds
        else:
            return None, None

