import torch
from typing import Union

from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.rssm.objectives.rssm_kl_objective import RSSMKLObjective
from ssm_rl.rssm.objectives.rssm_objective import RSSMObjective
from ssm_rl.model_learning.abstract_objective_factory import AbstractModelObjectiveFactory
import ssm_rl.model_learning.objective_factories as obj_fac


nn = torch.nn


class RSSMKLObjectiveFactory(obj_fac.AbstractKLObjectiveFactory):

    @staticmethod
    def build(distribution: str,
              config: ConfigDict):
        kwargs = RSSMKLObjectiveFactory.config_to_kwargs(config=config)
        return RSSMKLObjective(distribution=distribution, **kwargs)


class RSSMObjectiveFactory(AbstractModelObjectiveFactory):

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = super(RSSMObjectiveFactory, self).get_default_config(finalize_adding=False)
        config.add_subconf("kl", RSSMKLObjectiveFactory.get_default_config())
        if finalize_adding:
            config.finalize_adding()
        return config

    def _build(self,
               model,
               target_sizes: list[Union[int, tuple[int, int, int]]],
               target_types: list[str],
               bypass_mask: list[bool],
               config: ConfigDict,
               device):
        obs_objectives, reward_objective, inverse_dynamics_objective = super()._build(model=model,
                                                                                      target_sizes=target_sizes,
                                                                                      target_types=target_types,
                                                                                      bypass_mask=bypass_mask,
                                                                                      config=config,
                                                                                      device=device)

        kl_objective = RSSMKLObjectiveFactory.build(distribution=model.transition_model.latent_distribution,
                                                    config=config.kl)

        return obs_objectives, reward_objective, kl_objective, inverse_dynamics_objective

    def build(self,
              model,
              target_sizes: list[Union[int, tuple[int, int, int]]],
              target_types: list[str],
              bypass_mask: list[bool],
              config: ConfigDict,
              device):
        observation_objectives, reward_objective, kl_objective, inverse_dynamics_objective = \
            self._build(model=model,
                        target_sizes=target_sizes,
                        target_types=target_types,
                        bypass_mask=bypass_mask,
                        config=config,
                        device=device)
        return RSSMObjective(model=model,
                             obs_objectives=observation_objectives,
                             reward_objective=reward_objective,
                             kl_objective=kl_objective,
                             inverse_dyn_objective=inverse_dynamics_objective)
