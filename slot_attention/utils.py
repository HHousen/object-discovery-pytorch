from typing import Any, Tuple, Union

import torch
from pytorch_lightning import Callback

import wandb


def conv_transpose_out_shape(
    in_size, stride, padding, kernel_size, out_padding, dilation=1
):
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + out_padding
        + 1
    )


def assert_shape(
    actual: Union[torch.Size, Tuple[int, ...]],
    expected: Tuple[int, ...],
    message: str = "",
):
    assert (
        actual == expected
    ), f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def rescale(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log(
                    {"images": [wandb.Image(images)]}, commit=False
                )


def to_rgb_from_tensor(x: torch.Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = torch.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-1)
    return channels, masks
