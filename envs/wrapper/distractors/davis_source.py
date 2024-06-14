import os
from pathlib import Path
import numpy as np
import cv2

from envs.wrapper.distractors.abstract_source import AbstractDistractionSource

# - Videos are not long enough so there's random switches in the sequences
# - Images are downscaled weirdly
# - Fix set of sequences for easy and medium?
#
# Main advantage is that we can do both for and background only with davis -
# maybe segment more videos our selves using something that's trained on davis?

class DAVISDataSource(AbstractDistractionSource):
    DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)

    TRAINING_VIDEOS = [
        "bear",
        "bmx-bumps",
        "boat",
        "boxing-fisheye",
        "breakdance-flare",
        "bus",
        "car-turn",
        "cat-girl",
        "classic-car",
        "color-run",
        "crossing",
        "dance-jump",
        "dancing",
        "disc-jockey",
        "dog-agility",
        "dog-gooses",
        "dogs-scale",
        "drift-turn",
        "drone",
        "elephant",
        "flamingo",
        "hike",
        "hockey",
        "horsejump-low",
        "kid-football",
        "kite-walk",
        "koala",
        "lady-running",
        "lindy-hop",
        "longboard",
        "lucia",
        "mallard-fly",
        "mallard-water",
        "miami-surf",
        "motocross-bumps",
        "motorbike",
        "night-race",
        "paragliding",
        "planes-water",
        "rallye",
        "rhino",
        "rollerblade",
        "schoolgirls",
        "scooter-board",
        "scooter-gray",
        "sheep",
        "skate-park",
        "snowboard",
        "soccerball",
        "stroller",
        "stunt",
        "surf",
        "swing",
        "tennis",
        "tractor-sand",
        "train",
        "tuk-tuk",
        "upside-down",
        "varanus-cage",
        "walking",
    ]
    VALIDATION_VIDEOS = [
        "bike-packing",
        "blackswan",
        "bmx-trees",
        "breakdance",
        "camel",
        "car-roundabout",
        "car-shadow",
        "cows",
        "dance-twirl",
        "dog",
        "dogs-jump",
        "drift-chicane",
        "drift-straight",
        "goat",
        "gold-fish",
        "horsejump-high",
        "india",
        "judo",
        "kite-surf",
        "lab-coat",
        "libby",
        "loading",
        "mbike-trick",
        "motocross-jump",
        "paragliding-launch",
        "parkour",
        "pigs",
        "scooter-black",
        "shooting",
        "soapbox",
    ]

    def __init__(self, seed, shape, difficulty, data_path: str, train_or_val=None, intensity=1):
        super(DAVISDataSource, self).__init__(seed)

        self.shape = shape
        self.intensity = intensity
        data_path = Path(data_path)
        if self.check_empty(data_path / "DAVIS"):
            raise AssertionError("DAVIS data not found. Download it using the download script or adapt path...")
        path = data_path / "DAVIS" / "JPEGImages" / "480p"
        self.image_paths = self.get_img_paths(difficulty, path, train_or_val)
        self.num_path = len(self.image_paths)

        #self._cur_seq_idx = None
        self._idx = None
        self._cur_seq = None
        self._cur_mask_seq = None

        self.reset()

    def reset(self, eval_mode: bool = False):
        self._idx = 0
        cur_seq_idx = self._rng.randint(0, self.num_path)

        image_path = self.image_paths[cur_seq_idx]
        self._cur_seq = []
        self._cur_mask_seq = []
        num_imgs = len(os.listdir(image_path))
        for img_idx in range(num_imgs):
            fpath = image_path / "{:05d}.jpg".format(img_idx)
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            #img = np.asarray(Image.open(str(fpath)).resize(self.shape, resample=Image.Resampling.LANCZOS))
            img = img[:, :, ::-1]
            img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
            fpath = str(fpath)
            mpath = fpath.replace("JPEGImages", "Annotations_unsupervised").replace(
                "jpg", "png"
            )
            #mask = np.asarray(Image.open(str(mpath)).resize(self.shape, resample=Image.Resampling.LANCZOS))
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.shape[1], self.shape[0]))
            mask = np.logical_and(mask, True)
            self._cur_mask_seq.append(mask)
            self._cur_seq.append(img)

    def get_image(self):
        if self._idx == len(self._cur_seq):
            self.reset()

        img, mask = self._cur_seq[self._idx], self._cur_mask_seq[self._idx]
        self._idx += 1
        return img, mask

    def get_info(self):
        info = super().get_info()
        info["data_set"] = "DAVIS_2017"
        return info

    def ground_allowed(self, ground):
        return True

    def get_img_paths(self, difficulty: str, data_path: Path, train_or_val=None):
        num_frames = DAVISDataSource.DIFFICULTY_NUM_VIDEOS[difficulty]
        if train_or_val is None:
            dataset_images = sorted(data_path.iterdir())
        elif train_or_val in ["train", "training"]:
            dataset_images = DAVISDataSource.TRAINING_VIDEOS
        elif train_or_val in ["val", "validation"]:
            dataset_images = DAVISDataSource.VALIDATION_VIDEOS
        else:
            raise Exception(f"train_or_val {train_or_val} not defined.")

        image_paths = [data_path / subdir.name for subdir in dataset_images]
        if num_frames is not None:
            self._rng.shuffle(image_paths)
            if num_frames > len(image_paths) or num_frames < 0:
                raise ValueError(
                    f"`num_background_paths` is {num_frames} but should not be larger than the "
                    f"number of available background paths ({len(image_paths)}) and at least 0."
                )
            image_paths = image_paths[:num_frames]

        return image_paths

    def ignore_mask_for_background(self) -> bool:
        return True

    @staticmethod
    def check_empty(path: Path) -> bool:
        return not path.exists() or not any(path.iterdir())