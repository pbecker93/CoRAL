import torch
from typing import Optional
from torchvision.transforms import RandomCrop, CenterCrop

F = torch.nn.functional


class AbstractImgPreprocessor(torch.nn.Module):

    def __init__(self, input_img_size: tuple[int, int]):
        super().__init__()
        self._input_img_size = input_img_size

    def forward(self, img: torch.Tensor, eval: bool = False) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_img_size(self) -> tuple[int, int]:
        return self._input_img_size


class ImgPreprocessor(AbstractImgPreprocessor):

    def forward(self, img: torch.Tensor, eval: bool = False) -> torch.Tensor:
        img = img.float()
        img.div_(255.0).sub_(0.5)
        return img


class ColorDepthReductionImgPreprocessor(AbstractImgPreprocessor):

    __constants__ = ["_depth_bits", "_add_cb_noise"]
    _depth_bits: int
    _add_cb_noise: bool

    def __init__(self,
                 input_img_size: tuple[int, int],
                 depth_bits: int,
                 add_cb_noise: bool):

        super(ColorDepthReductionImgPreprocessor, self).__init__(input_img_size=input_img_size)
        assert 1 <= depth_bits <= 8
        self._depth_bits = depth_bits
        self._add_cb_noise = add_cb_noise

    def forward(self, img: torch.Tensor, eval: bool = False):
        img = img.float()
        img.div_(2 ** (8 - self._depth_bits)).floor_().div_(2 ** self._depth_bits).sub_(0.5)
        if self._add_cb_noise:
            img.add_(torch.rand_like(img).div_(2 ** self._depth_bits))
        return img


class ShiftImgPreprocessor(AbstractImgPreprocessor):

    __constants__ = ["pad"]

    def __init__(self, input_img_size: tuple[int, int], pad: int):
        super(ShiftImgPreprocessor, self).__init__(input_img_size=input_img_size)
        self.pad = pad

    def forward(self, img: torch.Tensor, eval: bool = False) -> torch.Tensor:
        img = img.float()
        img.div_(255.0).sub_(0.5)
        if eval:
            return img
        else:
            """Random Shift Augmentation from DRQv2 [2].
            Implementation can be found in [3].
            Adjusted to handle time series by 'retracted'.
            [2] I. Kostrikov, D. Yarats, and R. Fergus,
            “Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels,”
            arXiv:2004.13649 [cs, eess, stat], Mar. 2021, Accessed: Dec. 13, 2021. [Online].
            Available: http://arxiv.org/abs/2004.13649
            [3] https://github.com/facebookresearch/drqv2/blob/21e9048bf59e15f1018b49b850f727ed7b1e210d/drqv2.py#L14
            """
            n, t, c, h, w = img.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(img.reshape(n * t, c, h, w), padding, "replicate")

            eps = 1.0 / (h + 2 * self.pad)
            arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, t, 1, 1, 1)

            shift = torch.randint(
                0, 2 * self.pad + 1, size=(n, 1, 1, 1, 2), device=x.device, dtype=x.dtype
            )
            shift *= 2.0 / (h + 2 * self.pad)

            grid = base_grid + shift
            sampled = F.grid_sample(
                x, grid.reshape(n * t, w, h, 2), padding_mode="zeros", align_corners=False
            )
            return sampled.reshape(n, t, c, h, w)


class CropImagePreprocessor(AbstractImgPreprocessor):

    __constants__ = ["crop_size"]

    def __init__(self, input_img_size: tuple[int, int], crop_size: int):
        super(CropImagePreprocessor, self).__init__(input_img_size=input_img_size)
        self.crop_size = crop_size

    def forward(self, img: torch.Tensor, eval: bool = False) -> torch.Tensor:
        """
        From CORE Implementation:
         https://github.com/apple/ml-core/blob/ca00d2df2f24c0e3a52993e94d4b08887d9fe88b/utils.py
        """
        img = img.float()
        img.div_(255.0).sub_(0.5)
        assert len(img.shape) >= 3
        channels, height, width = img.shape[-3], img.shape[-2], img.shape[-1]
        if not eval:
            transform = RandomCrop((self.crop_size, self.crop_size), padding=0, padding_mode='edge')
            orig_shape = list(img.shape[:-2])
            if len(img.shape) >= 5:
                time_steps = img.shape[-4]
                channels = channels * time_steps
            obs = img.view(-1, channels, height, width)
            cropped_obs = torch.zeros(obs.size(0), channels, self.crop_size, self.crop_size, dtype=obs.dtype,
                                      device=obs.device)
            for i in range(obs.size(0)):
                cropped_obs[i, ...] = transform(obs[i, ...])
            cropped_obs = cropped_obs.view(*orig_shape, self.crop_size, self.crop_size)
        else:
            transform = CenterCrop((self.crop_size, self.crop_size))
            cropped_obs = transform(img)
        return cropped_obs

    @property
    def output_img_size(self) -> tuple[int, int]:
        return self.crop_size, self.crop_size
