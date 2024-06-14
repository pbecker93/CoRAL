from typing import Tuple

import numpy as np
import os

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations

from PIL import Image


class PreRenderedSource(AbstractDistractionSource):

    def __init__(self,
                 seed: int,
                 shape: tuple[int, int],
                 vid_folder: str,
                 max_seq_length: int,
                 temp_step: int = 5):
        #assert shape[0] == shape[1] == 64

        super(PreRenderedSource, self).__init__(seed=seed)

        self._all_videos = [os.path.join(vid_folder, vid) for vid in os.listdir(vid_folder)]
        self._temp_step = temp_step
        self._max_seq_length = max_seq_length

        self._shape = shape
        self._current_frames = None
        self._frame_idx = None

    def reset(self, eval_mode: bool = False):
        cur_vid_idx = self._rng.randint(low=0, high=len(self._all_videos))
        self._current_frames = dict(np.load(self._all_videos[cur_vid_idx]))["frames"]

        high = self._current_frames.shape[0] - self._max_seq_length * self._temp_step
        self._frame_idx = self._rng.randint(low=0, high=high)

    def get_image(self) -> Tuple[np.array, np.array]:
        img = self._current_frames[self._frame_idx]
        if self._shape[0] != img.shape[0] or self._shape[1] != img.shape[1]:
            img = np.array(Image.fromarray(img).resize(self._shape))
        self._frame_idx += self._temp_step
        mask = np.any(img > 14, axis=2)
        return img, mask

    def ground_allowed(self, ground):
        return ground in [DistractionLocations.FOREGROUND, DistractionLocations.BACKGROUND]