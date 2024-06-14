from ssm_rl.common.abstract_evaluator_factory import AbstractEvaluatorFactory
from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy
from ssm_rl.policy_learning.evaluation.reward_evaluator import RewardEvaluator
from ssm_rl.policy_learning.evaluation.collection_evaluator import CollectionEvaluator


def get_policy_eval_factory_from_string(eval_name: str) -> AbstractEvaluatorFactory:
    if eval_name == "reward":
        return RewardEvalFactory()
    elif eval_name == "collection":
        return CollectionEvaluatorFactory()
    else:
        raise ValueError(f"Unknown evaluator name: {eval_name}.")


class RewardEvalFactory(AbstractEvaluatorFactory):

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.num_sequences = 10
        config.use_mean = True
        config.max_sequence_length = -1
        config.record_vid = False
        config.eval_interval = 20
        config.log_info = None
        config.render_kwargs = None

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(
              env,
              policy: AbstractPolicy,
              save_path: str,
              config: ConfigDict):
        return RewardEvaluator(env=env,
                               policy=policy,
                               num_eval_sequences=config.num_sequences,
                               use_map=True,
                               eval_at_mean=config.use_mean,
                               max_sequence_length=config.max_sequence_length,
                               log_info=config.log_info,
                               eval_interval=config.eval_interval,
                               record_eval_vid=config.record_vid,
                               save_path=save_path,
                               record_kwargs=config.render_kwargs.copy() if config.render_kwargs is not None else None)

    @staticmethod
    def name() -> str:
        return RewardEvaluator.name()


class CollectionEvaluatorFactory(AbstractEvaluatorFactory):

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.num_sequences = 100
        config.use_map = True
        config.use_mean = True
        config.collection_interval = 20
        config.regenerate_obs = False
        config.collection_obs_type = "img_pro"

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(
              env,
              policy: AbstractPolicy,
              save_path: str,
              config: ConfigDict):
        return CollectionEvaluator(env=env,
                                   policy=policy,
                                   save_path=save_path,
                                   num_sequences=config.num_sequences,
                                   collect_at_map=config.use_map,
                                   collect_at_mean=config.use_mean,
                                   interval=config.collection_interval,
                                   regenerate_obs=config.regenerate_obs,
                                   collection_obs_type=config.collection_obs_type,
                                   save_original_obs=True)

    @staticmethod
    def name() -> str:
        return CollectionEvaluator.name()
