import torch

from ssm_rl.model_learning.abstract_objective import AbstractModelObjective

nn = torch.nn
data = torch.utils.data


class AbstractModelEvaluator(nn.Module):

    def __init__(self,
                 model_objective: AbstractModelObjective,
                 save_path: str):
        super(AbstractModelEvaluator, self).__init__()
        self._save_path = save_path
        self._model_objective = model_objective
        self._model = model_objective.model

    def evaluate(self,
                 data_loader: data.DataLoader,
                 iteration: int) -> dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def will_evaluate(self, iteration: int):
        pass
