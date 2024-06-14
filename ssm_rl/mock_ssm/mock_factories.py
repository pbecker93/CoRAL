from typing import Union
from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.mock_ssm.mock_ssm import MockSSM
from ssm_rl.mock_ssm.mock_model_objective import MockObjective


class MockSSMFactory:

    def __init__(self):
        self.encoder_factories = []

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:

        config = ConfigDict()
        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              config: ConfigDict,
              input_sizes: list[Union[int, tuple[int, int, int]]],
              input_types: list[str],
              bypass_mask: list[bool],
              action_dim: int,
              with_obs_valid: bool):

        return MockSSM()


class MockSSMObjectiveFactory:

    def __init__(self):
        self.observation_loss_factories = []
        self.input_mask_generator_factories = None

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:

        config = ConfigDict()
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(model,
              target_sizes: list[Union[int, tuple[int, int, int]]],
              target_types: list[str],
              bypass_mask: list[bool],
              config: ConfigDict,
              device):
        return MockObjective()

    @staticmethod
    def will_build_mask_generator() -> bool:
        return False
