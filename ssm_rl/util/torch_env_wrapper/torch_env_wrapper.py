import gym
import torch
import numpy as np

from ssm_rl.util.torch_env_wrapper.torch_spaces import TorchBox, TorchTuple


class TorchEnvWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 dtype: torch.dtype = torch.float32):
        super(TorchEnvWrapper, self).__init__(env)
        self._dtype = dtype
        self._obs_are_images = env.obs_are_images

        self.action_space = TorchBox(low=torch.from_numpy(env.action_space.low).to(self._dtype),
                                     high=torch.from_numpy(env.action_space.high).to(self._dtype),
                                     shape=env.action_space.shape,
                                     dtype=self._dtype)

        obs_spaces = []
        for o, is_image in zip(env.observation_space, self._obs_are_images):
            if is_image:
                assert o.dtype == np.uint8, "Images need to be uint8"
                obs_spaces.append(TorchBox(low=torch.from_numpy(o.low),
                                           high=torch.from_numpy(o.high),
                                           shape=(o.shape[2], o.shape[0], o.shape[1]),
                                           dtype=torch.uint8))
            else:
                obs_spaces.append(TorchBox(low=torch.from_numpy(o.low).to(self._dtype),
                                           high=torch.from_numpy(o.high).to(self._dtype),
                                           shape=o.shape,
                                           dtype=self._dtype))
        self.observation_space = TorchTuple(obs_spaces)

    def get_type(self, dtype):
        if dtype == np.float32:
            return self._dtype
        elif dtype == np.float64:
            return self._dtype
        elif dtype == np.bool or dtype == bool:
            return torch.bool
        else:
            raise NotImplementedError

    def _obs_to_torch(self, obs: list[np.ndarray]) -> list[torch.Tensor]:
        torch_obs = []
        for o, is_image in zip(obs, self._obs_are_images):
            if is_image:
                torch_obs.append(torch.from_numpy(np.ascontiguousarray(np.transpose(o, axes=(2, 0, 1)))))
            else:
                torch_obs.append(torch.from_numpy(o).to(self._dtype))
        return torch_obs

    def _dict_to_torch(self, np_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        torch_dict = {}
        for k, v in np_dict.items():
            if k == "obs_valid":
                torch_dict[k] = [torch.BoolTensor([b]) for b in v]
            elif np.isscalar(v):
                torch_dict[k] = torch.Tensor([v]).to(self._dtype)
            elif isinstance(v, list):
                torch_dict[k] = [None if e is None else torch.Tensor(e).to(self.get_type(e.dtype)) for e in v]
            else:
                if len(v.shape) == 3 and v.shape[2] in [1, 3]:  # image
                    torch_dict[k] = torch.from_numpy(np.ascontiguousarray(np.transpose(v, axes=(2, 0, 1))))
                else:
                    torch_dict[k] = torch.from_numpy(v).to(self._dtype)
        return torch_dict

    def reset(self, *args, **kwargs):
        np_obs, np_info = self.env.reset(*args, **kwargs)
        return self._obs_to_torch(np_obs), self._dict_to_torch(np_info)

    def step(self, action: torch.Tensor):
        np_action = action.detach().cpu().numpy().astype(self.env.action_space.dtype)
        np_obs, scalar_reward, terminated, truncated, np_info = self.env.step(action=np_action)
        reward = torch.Tensor([scalar_reward]).to(self._dtype)
        terminated = torch.Tensor([terminated]).to(torch.bool)
        truncated = torch.Tensor([truncated]).to(torch.bool)
        return self._obs_to_torch(np_obs), reward, terminated, truncated, self._dict_to_torch(np_info)
