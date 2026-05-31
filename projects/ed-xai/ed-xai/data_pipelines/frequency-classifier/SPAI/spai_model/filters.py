# From: https://github.com/mever-team/spai
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch import fft
from torch import linalg


def filter_image_frequencies(
    image: torch.Tensor,
    mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filters the frequencies of an image according to the provided mask.

    :param image: The images to filter. Dimensionality: [[B] x C] x H x W
    :param mask: Circular frequency mask. 1 = allow, 0 = filter out.
    :return: (low_freq_image, high_freq_residual)
    """
    image = fft.fft2(image)
    image = fft.fftshift(image)

    filtered_image = image * mask
    residual_filtered_image = image * (1 - mask)

    filtered_image = fft.ifftshift(filtered_image)
    filtered_image = fft.ifft2(filtered_image).real
    residual_filtered_image = fft.ifftshift(residual_filtered_image)
    residual_filtered_image = fft.ifft2(residual_filtered_image).real

    return filtered_image, residual_filtered_image


def generate_circular_mask(
    input_size: int,
    mask_radius_start: int,
    mask_radius_stop: Optional[int] = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    coordinates = generate_centered_2d_coordinates_grid(input_size, device)
    radius = linalg.vector_norm(coordinates, dim=-1)
    mask = torch.where(radius < mask_radius_start, 1, 0)
    if mask_radius_stop is not None:
        mask = torch.where(radius > mask_radius_stop, 1, mask)
    return mask


def generate_centered_2d_coordinates_grid(
    size: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    assert size % 2 == 0, "Input size must be even."
    coords_values = torch.arange(0, size // 2, dtype=torch.float, device=device)
    coords_values = torch.cat([coords_values.flip(dims=(0,)), coords_values], dim=0)
    coordinates_x = coords_values.unsqueeze(dim=0).expand(size, -1)
    coordinates_y = torch.t(coordinates_x)
    coordinates = torch.stack([coordinates_x, coordinates_y], dim=2)
    return coordinates
