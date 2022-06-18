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
# - Remove unused code
# - Adapt code from JAX to PyTorch

"""Common losses used in training GANs and masked modeling."""
from typing import Optional

import torch
import torch.nn.functional as F


def squared_euclidean_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    b2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the pairwise squared Euclidean distance.

    Args:
        a: float32: (n, d): An array of points.
        b: float32: (m, d): An array of points.
        b2: float32: (d, m): b square transpose.

    Returns:
        d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
        a[i] and b[j].
    """
    if b2 is None:
        b2 = torch.sum(b.T**2, dim=0, keepdim=True)
    a2 = torch.sum(a**2, dim=1, keepdim=True)
    ab = torch.matmul(a, b.T)
    d = a2 - 2 * ab + b2
    return d


def entropy_loss(
    affinity: torch.Tensor,
    loss_type: str = "softmax",
    temperature: float = 1.0
) -> torch.Tensor:
    """Calculates the entropy loss."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(flat_affinity, axis=-1)
        onehots = F.one_hot(
            codes, flat_affinity.shape[-1]).to(codes)
        onehots = probs - (probs - onehots).detach()
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, axis=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss
