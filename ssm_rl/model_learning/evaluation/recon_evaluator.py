import torch
import os
import matplotlib.pyplot as plt
import time

from ssm_rl.model_learning.evaluation.abstract_model_evaluator import AbstractModelEvaluator
from ssm_rl.model_learning.abstract_objective import AbstractModelObjective

nn = torch.nn


class ReconstructionEvaluator(AbstractModelEvaluator):

    def __init__(self,
                 model_objective: AbstractModelObjective,
                 conv_net: nn.Module,
                 save_path: str,
                 reconstruct_from_prior: bool = False,
                 save_num_sequences: int = 3,
                 save_imgs_every: int = 20,
                 fixed_eval_loader=None):
        super(ReconstructionEvaluator, self).__init__(model_objective=model_objective,
                                                      save_path=save_path)
        self._conv_net = conv_net

        self._opt = torch.optim.Adam(self._conv_net.parameters(), lr=1e-3)
        self._reconstruct_from_prior = reconstruct_from_prior

        self._save_path = os.path.join(save_path, "recon_eval")
        self._save_num_sequences = save_num_sequences
        self._save_imgs_every = save_imgs_every
        os.makedirs(self._save_path, exist_ok=True)
        self._fixed_eval_loader = fixed_eval_loader

    def evaluate(self,
                 data_loader: torch.utils.data.DataLoader,
                 iteration: int) -> dict[str, float]:
        loss_accu = 0.0
        step_count = 0
        t0 = time.time()
        for batch in data_loader:
            obs, actions, _, _, _, info = batch
            obs_valid = info.get("obs_valid", None)
            loss_masks = info.get("loss_masks", None)
            with torch.no_grad():

                assert obs_valid is None or all(ov is None for ov in obs_valid), NotImplementedError()
                assert loss_masks is None or all(lm is None for lm in loss_masks), NotImplementedError()

                smoothed_or_post_states, prior_states = \
                    self._model_objective.compute_states(*batch,
                                                         return_prior_states=True,
                                                         smoothed_states_if_avail=True)
                targets = info["image"]
                if self._reconstruct_from_prior:
                    latent_features = self._model.get_features(prior_states)
                    latent_features = latent_features[:, 1:]
                    targets = targets[:, 1:]
                else:
                    latent_features = self._model.get_features(smoothed_or_post_states)
            batch_size, seq_len, _ = latent_features.shape
            flat_latent_features = latent_features.reshape(-1, latent_features.shape[-1])

            flat_reconstructions = self._conv_net(flat_latent_features)
            reconstructions = flat_reconstructions.reshape(batch_size, seq_len, 3, 64, 64)

            loss = (reconstructions - targets).square().mean()

            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            loss_accu += loss.item()
            step_count += 1

        if iteration % self._save_imgs_every == 0 and iteration > 0:
            data_loader = data_loader if self._fixed_eval_loader is None else self._fixed_eval_loader
            for batch in data_loader:
                obs, actions, _, _, _, info = batch
                obs_valid = info.get("obs_valid", None)
                loss_masks = info.get("loss_masks", None)
                with torch.no_grad():

                    assert obs_valid is None or all(ov is None for ov in obs_valid), NotImplementedError()
                    assert loss_masks is None or all(lm is None for lm in loss_masks), NotImplementedError()

                    smoothed_or_post_states, prior_states = \
                        self._model_objective.compute_states(*batch,
                                                             return_prior_states=True,
                                                             smoothed_states_if_avail=True)
                    targets = info["image"]
                    if self._reconstruct_from_prior:
                        latent_features = self._model.get_features(prior_states)
                        latent_features = latent_features[:, 1:]
                        targets = targets[:, 1:]
                    else:
                        latent_features = self._model.get_features(smoothed_or_post_states)
                batch_size, seq_len, _ = latent_features.shape
                flat_latent_features = latent_features.reshape(-1, latent_features.shape[-1])

                flat_reconstructions = self._conv_net(flat_latent_features)
                reconstructions = flat_reconstructions.reshape(batch_size, seq_len, 3, 64, 64)

                for i in range(self._save_num_sequences):
                    cur_save_path = os.path.join(self._save_path, "{:05d}/{:02d}".format(iteration, i))
                    os.makedirs(cur_save_path, exist_ok=True)
                    start_idx = 0 # torch.randint(low=0, high=reconstructions.shape[1] - 10, size=(1,)).item()
                    for j in range(10):
                        r = ((reconstructions[i, start_idx + j].detach() + 0.5) * 255).to(torch.uint8).permute(1, 2, 0)
                        plt.imshow(r.cpu().numpy())
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(os.path.join(cur_save_path, "recon_{:02d}.png".format(j)), bbox_inches="tight")
                        plt.close()
                        #Image.fromarray(r.cpu().numpy()).save(os.path.join(cur_save_path, "recon_{:02d}.png".format(j)))

                        t = ((targets[i, start_idx + j] + 0.5) * 255).to(torch.uint8).permute(1, 2, 0)
                        plt.imshow(t.cpu().numpy())
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(os.path.join(cur_save_path, "target_{:02d}.png".format(j)), bbox_inches="tight")
                        plt.close()
                        #Image.fromarray(t.cpu().numpy()).save(os.path.join(cur_save_path, "target_{:02d}.png".format(j)))

                        o = ((obs[0][i, start_idx + j] + 0.5) * 255).to(torch.uint8).permute(1, 2, 0)
                        plt.imshow(o.cpu().numpy())
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(os.path.join(cur_save_path, "input_{:02d}.png".format(j)), bbox_inches="tight")
                        plt.close()
                        #Image.fromarray(o.cpu().numpy()).save(os.path.join(cur_save_path, "input_{:02d}.png".format(j)))

        return {"recon_loss": loss_accu / step_count, "time": time.time() - t0}

    def will_evaluate(self, iteration: int):
        return True

    @staticmethod
    def name() -> str:
        return "recon_eval"
