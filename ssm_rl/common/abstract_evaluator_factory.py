from ssm_rl.util.config_dict import ConfigDict


class AbstractEvaluatorFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:

        raise NotImplementedError

    @staticmethod
    def build(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @property
    def needs_image_in_info(self):
        return False

    @property
    def needs_state_in_info(self):
        return False
