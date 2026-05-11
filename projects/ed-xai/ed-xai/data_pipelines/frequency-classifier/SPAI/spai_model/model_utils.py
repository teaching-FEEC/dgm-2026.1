# From: https://github.com/mever-team/spai
# SPDX-License-Identifier: Apache-2.0

import torch


def patchify_image(
    img: torch.Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int]
) -> torch.Tensor:
    """Splits an input image into patches.

    :param img: Input image of size (B, C, H, W).
    :param patch_size: (height, width) of patches.
    :param stride: Stride on (height, width) dimensions.
    :returns: Patchified image of size (B, L, C, patch_height, patch_width).
    """
    kh, kw = patch_size
    dh, dw = stride
    img = img.unfold(2, kh, dh).unfold(3, kw, dw)
    img = img.permute(0, 2, 3, 1, 4, 5)
    img = img.contiguous()
    img = img.view(img.size(0), -1, img.size(3), kh, kw)
    return img
