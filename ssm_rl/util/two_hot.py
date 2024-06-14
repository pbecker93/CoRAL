import torch


class TwoHotEncoding(torch.nn.Module):

    def __init__(self,
                 lower_boundary: float,
                 upper_boundary: float,
                 num_bins: int):
        super().__init__()
        self._lower_boundary = lower_boundary
        self._upper_boundary = upper_boundary
        self._num_bins = num_bins
        self._bin_width = (self._upper_boundary - self._lower_boundary) / (num_bins - 1)
        bin_centers = torch.linspace(self._lower_boundary, self._upper_boundary, num_bins).unsqueeze(0)
        self.register_buffer("_bin_centers", bin_centers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #assert torch.all(self._lower_boundary <= x) and torch.all(x <= self._upper_boundary)
        #assert x.shape[-1] == 1
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, 1)
        idx = (x - self._lower_boundary) / self._bin_width
        lower_bin_idx = torch.floor(idx).long()
        lower_centers = torch.gather(self._bin_centers.T, 0, lower_bin_idx)
        # We need 1 - relative distance, this is equivalent to swapping the order of the two values
        lower_values = (lower_centers + self._bin_width - x).abs() / self._bin_width
        upper_values = (lower_centers - x).abs() / self._bin_width
        two_hot = torch.zeros(size=(x.shape[0], self._num_bins), device=x.device, dtype=x.dtype)
        two_hot.scatter_(-1, lower_bin_idx, lower_values)
        two_hot.scatter_(-1, lower_bin_idx + 1, upper_values)
        return two_hot.reshape(orig_shape + (self._num_bins, ))

    def decode(self, two_hot: torch.Tensor) -> torch.Tensor:
        #assert two_hot.shape[-1] == self._num_bins
        #assert torch.allclose(two_hot.sum(dim=-1), torch.ones(two_hot.shape[0],
        #                                                      device=two_hot.device,
        #                                                       dtype=two_hot.dtype))
        return torch.sum(two_hot * self._bin_centers, dim=-1, keepdim=True)


if __name__ == "__main__":

    device = torch.device("cuda")

    _lower = -20
    _upper = 20
    _num_bins = 255
    _two_hot_encoding = TwoHotEncoding(lower_boundary=_lower,
                                       upper_boundary=_upper,
                                       num_bins=_num_bins).to(device)

    _x = (_upper - _lower) * torch.rand(size=(100, 1)) + _lower
    _x = _x.to(device)
    _two_hot = _two_hot_encoding.encode(_x)
    _y = _two_hot_encoding.decode(_two_hot)
    print("bla")

