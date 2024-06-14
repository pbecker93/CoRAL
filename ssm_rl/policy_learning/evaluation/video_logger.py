import imageio
import os
import pathlib
from PIL import Image
import warnings
import numpy as np
from typing import Optional


class VideoLogger:

    def __init__(self,
                 save_path: str = None,
                 to_wandb: bool = True,
                 fps: int = 30,
                 save_single_images: bool = False,
                 render_kwargs: Optional[dict] = None):
        self.reset()
        self._save_path = save_path
        self._to_wandb = to_wandb
        self._fps = fps
        self._buffer = None
        self._save_single_images = save_single_images
        self._render_kwargs = {} if render_kwargs is None else render_kwargs
        if self._to_wandb:
            import wandb
            self._wandb = wandb

    def reset(self):
        self._buffer = []

    def record_env(self, env):
        self._buffer.append(env.render(**self._render_kwargs))

    def record_obs(self, obs):
        self._buffer.append(np.transpose(obs.cpu().numpy(), (1, 2, 0)))

    def save(self, file_name, step):
        name = "{}_{:05d}".format(file_name, step)

        if self._save_path is not None:
            path = os.path.join(self._save_path, "{}.mp4".format(name))
            if self._save_single_images:
                pathlib.Path(path[:-4]).mkdir(parents=False, exist_ok=True)
                for i, img in enumerate(self._buffer):
                    Image.fromarray(img).save(os.path.join(path[:-4], "{:03d}.png".format(i)))
            imageio.mimsave(path, self._buffer, fps=self._fps)
        if self._to_wandb:
            try:
                img_array = np.transpose(np.stack(self._buffer, axis=0), (0, 3, 1, 2))
                self._wandb.log({"eval_vid": self._wandb.Video(img_array, fps=self._fps)}, step=step)
            except self._wandb.errors.Error:
                warnings.warn("WandB does not seem to be initialized! - Ignoring WandB logging")
