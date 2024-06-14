import matplotlib
matplotlib.use("Agg")
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from ssm_rl.model_learning.evaluation.abstract_model_evaluator import AbstractModelEvaluator
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective

nn = torch.nn


class SaliencyEvaluator(AbstractModelEvaluator):

    def __init__(self,
                 model_objective: AbstractModelObjective,
                 save_path: str,
                 save_num_images: int = 10,
                 eval_interval: int = 50,
                 sub_seq_length: int = 8,
                 obs_idx: int = 0,
                 fixed_eval_loader=None):
        super(SaliencyEvaluator, self).__init__(model_objective=model_objective,
                                                save_path=save_path)
        self._obs_idx = obs_idx

        self._save_path = os.path.join(save_path, "saliency")
        self._save_num_images = save_num_images
        self._eval_interval = eval_interval

        self._sub_seq_length = sub_seq_length
        os.makedirs(self._save_path, exist_ok=True)
        self._fixed_eval_loader = fixed_eval_loader

    def evaluate(self,
                 data_loader: torch.utils.data.DataLoader,
                 iteration: int) -> dict[str, float]:
        t0 = time.time()
        if self.will_evaluate(iteration):
            data_loader = data_loader if self._fixed_eval_loader is None else self._fixed_eval_loader
            cur_save_path = os.path.join(self._save_path, "{:05d}".format(iteration))
            os.makedirs(cur_save_path, exist_ok=True)
            for i, batch in enumerate(data_loader):

                if i < self._save_num_images:
                    #assert len(batch[-1]) == 0, NotImplementedError()
                    cur_batch = [
                        [o[0, :self._sub_seq_length] for o in batch[0]],  # obs
                        batch[1][0, :self._sub_seq_length],               # actions
                        batch[2][0, :self._sub_seq_length],               # rewards
                        batch[3][0, :self._sub_seq_length],               # terminated
                        batch[4][0, :self._sub_seq_length],               # truncated
                        {k: v[0, :self._sub_seq_length] for k, v in batch[5].items()}
                    ]

                    def func(obs_to_diff):
                        obs = cur_batch[0]
                        obs[self._obs_idx] = obs_to_diff
                        obs = [o.unsqueeze(0) for o in obs]
                        batch = [b.unsqueeze(0) for b in cur_batch[1:-1]]
                        batch = [obs] + batch + [{}]

                        smoothed_or_post_states = self._model_objective.compute_states(*batch,
                                                                                       return_prior_states=False,
                                                                                       smoothed_states_if_avail=True)
                        features = self._model.get_deterministic_features(smoothed_or_post_states)
                        return features.squeeze(0)

                    jac = torch.autograd.functional.jacobian(func, cur_batch[0][self._obs_idx])
                    jac_norm = jac.square().sum(dim=(0, 1)).sqrt()
                    jac_norms = torch.max(jac_norm, dim=1)[0].detach().cpu().numpy()

                    imgs = cur_batch[0][self._obs_idx].detach().cpu().numpy()

                    for img_idx in range(self._sub_seq_length):

                        plt.figure()
                        plt.imshow(np.transpose(imgs[img_idx], (1, 2, 0)) + 0.5)
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(os.path.join(cur_save_path, "img_{:02d}_{:02d}.png".format(i, img_idx)),
                                    bbox_inches="tight")
                        plt.close()
                        plt.figure()
                        plt.imshow(jac_norms[img_idx], cmap=plt.cm.gray)
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(os.path.join(cur_save_path, "saliency_{:02d}_{:02d}.png".format(i, img_idx)),
                                    bbox_inches="tight")
                        plt.close()

    def will_evaluate(self, iteration: int):
        return iteration % self._eval_interval == 0 and iteration > 0

    @staticmethod
    def name() -> str:
        return "saliency_eval"
