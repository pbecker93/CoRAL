import gym
import torch


class TorchBox(gym.spaces.Box):

    def __init__(self,
                 low: torch.Tensor,
                 high: torch.Tensor,
                 shape: tuple,
                 dtype: torch.dtype):
        self.low = low
        self.high = high
        self._shape = shape
        self.dtype = dtype

    def __getattr__(self, item):
        if item not in ["shape"]:
            raise NotImplementedError("Probably not implemented")
        return self.__getattribute__(item)


class TorchTuple(gym.spaces.Tuple):

    def __init__(self,
                 spaces):
        self.spaces = tuple(spaces)

    def __getattr__(self, item):
        raise NotImplementedError("Probably not implemented")
