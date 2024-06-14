import time
import torch
from typing import Optional, Union
from collections import OrderedDict

from ssm_rl.model_learning.abstract_objective import AbstractModelObjective
from ssm_rl.mock_ssm.mock_model_objective import MockObjective
from ssm_rl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_rl.policy_learning.common.abstract_policy import AbstractPolicy
from ssm_rl.policy_learning.common.entropy_bonus import EntropyBonus


data = torch.utils.data
nn = torch.nn
opt = torch.optim


class AbstractTrainer:

    def __init__(self,
                 model_objective: AbstractModelObjective,
                 model: AbstractSSM,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float):

        self.model_objective = model_objective
        self._model = model

        objective_params = self.model_objective.get_parameters_for_optimizer()
        if objective_params is None:
            self._model_optimizer, self._model_clip_fn = None, None
        else:
            self._model_optimizer, self._model_clip_fn = \
                self._build_optimizer_and_clipping(params=objective_params,
                                                   learning_rate=model_learning_rate,
                                                   adam_eps=model_adam_eps,
                                                   clip_norm=model_clip_norm,
                                                   weight_decay=model_weight_decay)


    def update_model_parameters(self,
                                model_loss: torch.Tensor,
                                retain_graph: bool = False):
        self._model_optimizer.zero_grad()
        model_loss.backward(retain_graph=retain_graph)
        self._model_clip_fn()
        self._model_optimizer.step()

    @staticmethod
    def _build_optimizer_and_clipping(params,
                                      learning_rate: float,
                                      adam_eps: float,
                                      clip_norm: float,
                                      weight_decay: float) -> tuple[opt.Optimizer, callable]:
        def clip_grads_if_desired(p):
            if clip_norm > 0:
                _ = nn.utils.clip_grad_norm_(p, clip_norm)
        if weight_decay > 0.0:
            optimizer = opt.AdamW(params, lr=learning_rate, eps=adam_eps, weight_decay=weight_decay)
        else:
            optimizer = opt.Adam(params=params, lr=learning_rate, eps=adam_eps)
        param_list = []
        for group in optimizer.param_groups:
            param_list.extend(list(group['params']))
        return optimizer, lambda: clip_grads_if_desired(p=param_list)

    def _train_on_batch(self, batch, env_step: int = -1, mode: Optional[str] = None) -> OrderedDict:
        raise NotImplementedError

    def train_epoch(self, data_loader: data.DataLoader, env_step: int = -1, mode: Optional[str] = None) -> tuple[OrderedDict, float]:
        batches_per_epoch = len(data_loader)
        avg_log_dict = None
        t0 = time.time()
        for i, batch in enumerate(data_loader):
            log_dict = self._train_on_batch(batch, env_step=env_step, mode=mode)
            if self.model_objective is not None:
                self.model_objective.post_gradient_step_callback()
            if i == 0:
                avg_log_dict = OrderedDict({k: v / batches_per_epoch for k, v in log_dict.items()})
            else:
                for k, v in log_dict.items():
                    avg_log_dict[k] += v / batches_per_epoch

        return avg_log_dict, time.time() - t0

    def get_optimizer_state_dict(self) -> dict:
        raise NotImplementedError

    def load_optimizer_state_dict(self, state_dict: dict):
        raise NotImplementedError


class AbstractMaxEntropyPolicyTrainer(AbstractTrainer):

    def __init__(self,
                 model_objective: AbstractModelObjective,
                 model: AbstractSSM,
                 policy: AbstractPolicy,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float,
                 entropy_bonus: float,
                 learnable_entropy_bonus: bool,
                 entropy_learning_rate: float,
                 entropy_adam_eps: float,
                 entropy_clip_norm: float,
                 entropy_exp_activation: bool,
                 target_entropy: Union[float, str] = "auto"):
        super().__init__(model_objective=model_objective,
                         model=model,
                         model_learning_rate=model_learning_rate,
                         model_adam_eps=model_adam_eps,
                         model_clip_norm=model_clip_norm,
                         model_weight_decay=model_weight_decay)
        self._policy = policy

        target_entropy = (- policy.action_dim) if target_entropy == "auto" else float(target_entropy)
        self._entropy_bonus = EntropyBonus(entropy_bonus=entropy_bonus,
                                           learnable=learnable_entropy_bonus,
                                           exp_activation=entropy_exp_activation,
                                           target_entropy=target_entropy).to(self._policy.device)
        if self._entropy_bonus.is_trainable:
            self._entropy_optimizer, self._entropy_clip_fn = \
                self._build_optimizer_and_clipping(params=self._entropy_bonus.parameters(),
                                                   learning_rate=entropy_learning_rate,
                                                   adam_eps=entropy_adam_eps,
                                                   clip_norm=entropy_clip_norm,
                                                   weight_decay=0.0)
