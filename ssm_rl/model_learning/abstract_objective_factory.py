import torch
from typing import Union

from ssm_rl.util.config_dict import ConfigDict
from ssm_rl.model_learning.nll_objective import NLLObjective

from ssm_rl.model_learning.objective_factories import (ReconstructionObjectiveFactory,
                                                       SSLObjectiveFactory,
                                                       InverseDynamicsObjectiveFactory)
from ssm_rl.model_learning.reconstruction_objective import ReconstructionObjective
nn = torch.nn


class AbstractModelObjectiveFactory:

    def __init__(self, observation_loss_factories):

        self.observation_loss_factories = observation_loss_factories

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        for i, factory in enumerate(self.observation_loss_factories):
            config.add_subconf("obs{}".format(i), factory.get_default_config())

        config.with_inv_dyn_loss = False
        config.add_subconf("inv_dyn", InverseDynamicsObjectiveFactory.get_default_config())

        config.reward_scale_factor = 1.0

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
        observation_objectives = []
        for i, (target_size, target_type, factory) in enumerate(zip(target_sizes, target_types,
                                                                    self.observation_loss_factories)):
            if not bypass_mask[i]:
                if isinstance(factory, ReconstructionObjectiveFactory) or factory == ReconstructionObjectiveFactory:
                    observation_objectives.append(factory.build(input_size=model.feature_size,
                                                                target_size=target_sizes[i],
                                                                target_type=target_types[i],
                                                                config=getattr(config, "obs{}".format(i))).to(device))
                elif isinstance(factory, SSLObjectiveFactory) or factory == SSLObjectiveFactory:
                    observation_objectives.append(factory.build(state_dim=model.feature_size,
                                                                obs_dim=model.transition_model.obs_sizes[i],
                                                                config=getattr(config, "obs{}".format(i))).to(device))
                else:
                    raise NotImplementedError

        model.reward_decoder.set_symlog(work_with_symlog=False)
        reward_objective = NLLObjective(scale_factor=config.reward_scale_factor).to(device)

        if config.with_inv_dyn_loss:
            inverse_dynamics_objective = InverseDynamicsObjectiveFactory.build(feature_dim=model.feature_size,
                                                                               action_dim=model.action_dim,
                                                                               config=config.inv_dyn).to(device)
        else:
            inverse_dynamics_objective = None
        obs_objectives = nn.ModuleList(observation_objectives)

        encoder_symlog_mask = [isinstance(o, ReconstructionObjective) and o.usees_symlog for o in obs_objectives]
        [e.set_symlog(work_with_symlog=use_symlog) for e, use_symlog in zip(model.encoders, encoder_symlog_mask)]
        return obs_objectives, reward_objective, inverse_dynamics_objective

    def build(self,
              model,
              target_sizes: list[Union[int, tuple[int, int, int]]],
              target_types: list[str],
              bypass_mask: list[bool],
              config: ConfigDict,
              device):
        raise NotImplementedError
