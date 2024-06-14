import torch

from ssm_rl.util.config_dict import ConfigDict
import ssm_rl.util.normalizer as ret_norm


data = torch.utils.data
nn = torch.nn
opt = torch.optim


class AbstractTrainerFactory:

    def get_default_config(self) -> ConfigDict:
        config = ConfigDict()
        config.model_learning_rate = 6e-4
        config.model_adam_epsilon = 1e-8
        config.model_clip_norm = 100.0
        config.model_weight_decay = 0.0
        return config

    def __init__(self,
                 objective_factory):
        self.objective_factory = objective_factory


class AbstractMaxEntropyPolicyTrainerFactory(AbstractTrainerFactory):

    def get_default_config(self) -> ConfigDict:
        config = super().get_default_config()

        config.add_subconf("entropy", ConfigDict())

        config.entropy.learning_rate = 3e-4
        config.entropy.adam_epsilon = 1e-8
        config.entropy.clip_norm = 1.0
        config.entropy.exp_activation = True
        config.entropy.target = "auto"
        config.entropy.bonus = 0.0
        config.entropy.learn_bonus = False
        return config
