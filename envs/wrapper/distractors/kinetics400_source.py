from pathlib import Path
import numpy as np
import warnings
import skvideo.io
import cv2

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource
from envs.wrapper.merge_strategy.merge_strategies import DistractionLocations


class Kinetics400DataSource(AbstractDistractionSource):

    DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)

    def __init__(self, seed, shape, difficulty, data_path: str, grayscale: bool, seq_length: int):

        super(Kinetics400DataSource, self).__init__(seed)

        self.shape = shape
        self.grayscale = grayscale
        self.seq_length = seq_length
        data_path = Path(data_path)
        if self.check_empty(data_path / "kinetics400"):
            raise AssertionError("kinetics400 data not found. Download it using the download script or adapt path...")

        path = data_path / "kinetics400"
        self.train_images_paths, self.val_images_paths = self.get_img_paths(difficulty, path)

        self._idx = None
        self.num_images = None
        self.bg_arr = None
        self.mask_arr = None

    def get_img_paths(self, difficulty, data_path: Path, train_or_val=None):
        num_training_files = Kinetics400DataSource.DIFFICULTY_NUM_VIDEOS[difficulty]
        train_images = (data_path / "train").glob("*.mp4")
        val_images = (data_path / "test").glob("*.mp4")

        train_image_paths = list(train_images)
        if num_training_files is not None:
            if num_training_files > len(train_image_paths) or num_training_files < 0:
                raise ValueError(
                    f"`num_background_paths` is {num_training_files} but should not be larger than the "
                    f"number of available background paths ({len(train_image_paths)}) and at least 0."
                )
            train_image_paths = train_image_paths[:num_training_files]

        val_image_paths = list(val_images)

        return train_image_paths, val_image_paths

    def get_info(self):
        info = {}
        info["data_set"] = "KINETICS_400"
        return info

    def read_in_file(self, fname, grayscale=False):
        with warnings.catch_warnings():
            # Filter DeprecationWarning from skvideo (ffmpeg)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            if grayscale:
                frames = skvideo.io.vread(str(fname),
                                          num_frames=self.seq_length,
                                          outputdict={"-pix_fmt": "gray"})
            else:
                frames = skvideo.io.vread(str(fname),
                                          num_frames=self.seq_length)
            img_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1], 1) if self.grayscale else
                               (frames.shape[0], self.shape[0], self.shape[1], 3),
                               dtype=np.uint8)
            for i in range(frames.shape[0]):
                # THIS IS NOT A BUG! cv2 uses (width, height)
                resized_img = cv2.resize(frames[i], (self.shape[1], self.shape[0]))
                img_arr[i] = resized_img if len(resized_img.shape) == 3 else np.expand_dims(resized_img, axis=-1)
            return img_arr

    def get_image(self) -> tuple[np.array, np.array]:
        if self._idx == len(self.bg_arr):
            raise AssertionError("No more images available. Did you call reset()?")

        img, mask = self.bg_arr[self._idx], self.mask_arr[self._idx]
        self._idx += 1
        return img, mask

    def reset(self, eval_mode: bool = False):
        self._idx = 0
        paths = self.val_images_paths if eval_mode else self.train_images_paths
        loaded = False
        while not loaded:
            idx = self._rng.randint(0, len(paths))
            fname = paths[idx]
            print("Loading Video: ", fname, "form {}".format("val" if eval_mode else "train"))
            try:
                img_arr = self.read_in_file(fname, grayscale=self.grayscale)
                loaded = True
            except RuntimeError as e:
                print("Could not load video: ", fname, "form {}".format("val" if eval_mode else "train"))
                print("Error: ", e)
                pass

        self.num_images = len(img_arr)
        self.bg_arr = img_arr
        self.mask_arr = np.full(img_arr.shape[:-1], True)

    @staticmethod
    def check_empty(path: Path) -> bool:
        return not path.exists() or not any(path.iterdir())

    def ground_allowed(self, ground) -> bool:
        return ground == DistractionLocations.BACKGROUND
