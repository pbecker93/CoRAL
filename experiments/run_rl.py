from typing import Optional

from ssm_rl.util.config_dict import ConfigDict

from ssm_rl.rssm.rssm_factory import RSSMFactory

import ssm_rl.rssm.objectives.objective_factory as rssm_of
from ssm_rl.mock_ssm.mock_factories import MockSSMFactory, MockSSMObjectiveFactory

from ssm_rl.model_learning.objective_factories import SSLObjectiveFactory, ReconstructionObjectiveFactory
from ssm_rl.ssm_interface.encoder_factory import EncoderFactory
from ssm_rl.model_learning.evaluation.model_eval_factories import get_model_eval_factory_from_string
from ssm_rl.policy_learning.evaluation.policy_eval_factories import get_policy_eval_factory_from_string


from ssm_rl.policy_learning.latent_imagination.latent_imagination_policy_factory import LatentImaginationPolicyFactory
from ssm_rl.policy_learning.latent_imagination.latent_imagination_trainer_factory import \
    LatentImaginationPolicyTrainerFactory

from ssm_rl.policy_learning.actor_critic.actor_critic_policy_factory import SACPolicyFactory
from ssm_rl.policy_learning.actor_critic.sac_trainer_factory import SACPolicyTrainerFactory
from ssm_rl.policy_learning.evaluation.policy_eval_factories import RewardEvalFactory
from ssm_rl.policy_learning.common.rl_factory import MBRLFactory

from experiments.env_factory import RLEnvFactory
from experiments.rl_experiment import RLExperiment


from experiments.cw_util.cw2_experiment import Cw2Experiment


class Cw2MBRLExperiment(Cw2Experiment):

    def __init__(self, *args, **kwargs):
        super(Cw2Experiment, self).__init__(*args, **kwargs)
        self._save_interval = None

    @property
    def save_interval(self) -> int:
        if self._save_interval is None:
            raise AssertionError("Experiment not initialized yet!")
        return self._save_interval

    @staticmethod
    def default_experiment_config():
        conf = ConfigDict(seed=0,
                          obs_type="img",
                          world_model="rssm",
                          model_objective="vi",
                          agent="sac",
                          save_interval=-1,
                          fully_deterministic=False,
                          model_eval=[],
                          policy_eval=[])
        conf.finalize_adding()
        return conf

    def setup_experiment(self,
                         exp_config: ConfigDict,
                         seed_offset: int = 0,
                         save_path: Optional[str] = None):

        # Basics
        env_factory = RLEnvFactory(exp_config.obs_type)
        mbrl_factory = MBRLFactory()
        self._save_interval = exp_config.save_interval
        num_obs = env_factory.get_num_obs()

        # Encoders
        encoder_factories = [EncoderFactory() for _ in range(num_obs)]

        # Observation Objectives
        if exp_config.model_objective in ["vi", "masked"]:
            observation_loss_factories = \
                [ReconstructionObjectiveFactory() for _ in range(num_obs)]
        elif exp_config.model_objective == "ssl":
            observation_loss_factories = [SSLObjectiveFactory() for _ in range(num_obs)]
        elif isinstance(exp_config.model_objective, list):
            #  assert len(exp_config.model_objective) == num_obs
            observation_loss_factories = []
            for objective_type in exp_config.model_objective:
                if objective_type == "vi":
                    observation_loss_factories.append(ReconstructionObjectiveFactory())
                elif objective_type == "ssl":
                    observation_loss_factories.append(SSLObjectiveFactory())
                else:
                    raise NotImplementedError(f"Unknown objective {objective_type}")
        elif exp_config.model_objective == "none":
            assert exp_config.world_model == "none", "Model objective must be none if world model is none!"
            observation_loss_factories = None
        else:
            raise ValueError(f"Unknown model_objective: {exp_config.model_objective}")

        # World Model and Model Objective
        if exp_config.world_model == "rssm":
            model_factory = RSSMFactory(encoder_factories=encoder_factories)
            model_objective_factory = \
                rssm_of.RSSMObjectiveFactory(observation_loss_factories=observation_loss_factories)
        elif exp_config.world_model == "none":
            model_factory = MockSSMFactory()
            model_objective_factory = MockSSMObjectiveFactory()
        else:
            raise ValueError(f"Unknown world model: {exp_config.world_model}")

        # Policy + Trainer
        if exp_config.agent == "li":
            policy_factory = LatentImaginationPolicyFactory()
            trainer_factory = LatentImaginationPolicyTrainerFactory(objective_factory=model_objective_factory)
        elif exp_config.agent == "sac":
            policy_factory = SACPolicyFactory()
            trainer_factory = SACPolicyTrainerFactory(objective_factory=model_objective_factory)
        else:
            raise ValueError(f"Unknown agent: {exp_config.agent}")

        # Evaluation
        model_eval_factories = [get_model_eval_factory_from_string(me) for me in exp_config.get("model_eval", [])]
        policy_eval_factories = [RewardEvalFactory()] + \
                                [get_policy_eval_factory_from_string(pe) for pe in exp_config.get("policy_eval", [])]

        return RLExperiment(env_factory=env_factory,
                            model_factory=model_factory,
                            policy_factory=policy_factory,
                            trainer_factory=trainer_factory,
                            rl_factory=mbrl_factory,
                            model_eval_factories=None if len(model_eval_factories) == 0 else model_eval_factories,
                            policy_eval_factories=policy_eval_factories,
                            save_path=save_path,
                            verbose=False,
                            seed=exp_config.seed + seed_offset,
                            use_cuda_if_available=True,
                            fully_deterministic=exp_config.fully_deterministic)


if __name__ == "__main__":
    import sys
    from cw2.cluster_work import ClusterWork
    # uncomment import for wandb logging - install wandb first
    # from experiments.cw_util.wb_logger import CustomWandBLogger

    from experiments.cw_util.printer_logger import PrintLogger

    if not any(".yml" in arg for arg in sys.argv):
        sys.argv.append("experiments/configs/manipulation_example.yml")
        sys.argv.append("-o")

    cw = ClusterWork(Cw2MBRLExperiment)
    cw.add_logger(PrintLogger(name="rl"))
    # uncomment for wandb logging
    # cw.add_logger(CustomWandBLogger())

    cw.run()

