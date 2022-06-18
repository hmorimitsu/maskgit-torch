# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by Henrique Morimitsu
# - Adapt code from JAX to PyTorch

"""Common layers and blocks."""

import functools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm_layer(
    filters: int,
    norm_type: str = 'BN'
) -> functools.partial:
    """Normalization layer."""
    if norm_type == 'BN':
        norm_fn = functools.partial(
            nn.BatchNorm2d,
            num_features=filters,
            momentum=0.9,
            eps=1e-5)
    elif norm_type == 'LN':
        norm_fn = functools.partial(nn.LayerNorm, normalized_shape=filters)
    elif norm_type == 'GN':
        norm_fn = functools.partial(nn.GroupNorm, num_groups=32, num_channels=filters)
    else:
        raise NotImplementedError
    return norm_fn


class Upsample(nn.Module):
    def __init__(
        self,
        factor: float = 2
    ) -> None:
        super().__init__()
        self.factor = factor

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.factor, mode='nearest')


class Downsample(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (2, 2),
        stride: Tuple[int, int] = (2, 2)
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Simple average pooling.

        Notice the JAX version uses a more precise pooling.
        It has not been tested how much the change of pooling affect the results.

        Args:
        x: Input tensor

        Returns:
        pooled: Tensor after applying pooling.
        """
        return F.avg_pool2d(x, self.kernel_size, self.stride)
