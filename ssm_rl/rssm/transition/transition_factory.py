from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.rssm.transition.r_rssm_tm import RRSSMTM
from ssm_rl.rssm.transition.cat_rssm_tm import CatRSSMTM

class TransitionFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = False) -> ConfigDict:
        config = ConfigDict()
        config.type = "r_rssm"
        config.lsd = 30
        config.rec_state_dim = 200
        config.num_layers = 1
        config.layer_size = 200
        config.min_std = 0.1
        config.activation = "ReLU"

        # Monte Carlo Dropout
        # Categorical
        config.num_categoricals = 32
        config.categorical_size = 32

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(config: ConfigDict,
              obs_sizes: list[int],
              action_dim: int,
              with_obs_valid: bool):
        if "cat" in config.type:
            return CatRSSMTM(obs_sizes=obs_sizes,
                             categorical_size=config.categorical_size,
                             num_categorical=config.num_categoricals,
                             action_dim=action_dim,
                             rec_state_dim=config.rec_state_dim,
                             num_layers=config.num_layers,
                             layer_size=config.layer_size,
                             with_obs_pre_layers=config.with_obs_pre_layers,
                             build_with_obs_valid=with_obs_valid)
        else:
            default_params = {"obs_sizes": obs_sizes,
                              "build_with_obs_valid": with_obs_valid,
                              "state_dim": config.lsd,
                              "action_dim": action_dim,
                              "num_layers": config.num_layers,
                              "layer_size": config.layer_size,
                              "min_std": config.min_std,
                              "activation": config.activation,
                              "rec_state_dim": config.rec_state_dim}
            return RRSSMTM(**default_params)
