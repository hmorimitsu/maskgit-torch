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
# - Add comments about ResBlock

r"""MaskGIT Tokenizer based on VQGAN.

This tokenizer is a reimplementation of VQGAN [https://arxiv.org/abs/2012.09841]
with several modifications. The non-local layers are removed from VQGAN for
faster speed.
"""
from typing import Any, Dict, Tuple

from einops import rearrange
import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskgit.libml import losses
from maskgit.nets import layers


class ResBlock(nn.Module):
    """Basic Residual Block."""
    def __init__(
        self,
        input_dim: int,
        filters: int,
        norm_type: str,
        conv_fn: Any,
        activation_fn: Any = nn.ReLU,
        use_conv_shortcut: bool = False
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn

        self.norm0 = layers.get_norm_layer(input_dim, norm_type)()
        self.conv0 = conv_fn(input_dim, filters, kernel_size=3, padding=1, bias=False)
        self.norm1 = layers.get_norm_layer(filters, norm_type)()
        self.conv1 = conv_fn(filters, filters, kernel_size=3, padding=1, bias=False)

        self.conv_res = None
        if input_dim != filters:
            kernel_size = 3 if use_conv_shortcut else 1
            # TODO: the original code does not have a residual path, it is probably a bug
            # self.conv_res = conv_fn(
            #   input_dim, filters, kernel_size=kernel_size, bias=False)
            self.conv_res = conv_fn(
                filters, filters, kernel_size=kernel_size, bias=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.norm0(x)
        x = self.activation_fn(x)
        x = self.conv0(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        if self.conv_res is not None:
            # TODO: this is not residual, it is reusing the x from the same path
            # residual = self.conv_res(residual)
            residual = self.conv_res(x)
        return x + residual


class Encoder(nn.Module):
    """Encoder Blocks."""
    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ) -> None:
        super().__init__()

        input_dim = config.vqvae.input_dim
        filters = config.vqvae.filters
        num_res_blocks = config.vqvae.num_res_blocks
        channel_multipliers = config.vqvae.channel_multipliers
        embedding_dim = config.vqvae.embedding_dim
        conv_downsample = config.vqvae.conv_downsample
        norm_type = config.vqvae.norm_type
        if config.vqvae.activation_fn == "relu":
            self.activation_fn = F.relu
        elif config.vqvae.activation_fn == "swish":
            self.activation_fn = lambda x: x * torch.sigmoid(x)
        else:
            raise NotImplementedError

        conv_fn = nn.Conv2d
        block_args = dict(
            norm_type=norm_type,
            conv_fn=conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )

        self.conv_in = conv_fn(input_dim, filters, kernel_size=3, padding=1, bias=False)

        num_blocks = len(channel_multipliers)
        self.res_blocks = nn.ModuleList()
        prev_filters = filters
        for i in range(num_blocks):
            block_layers = nn.ModuleList()
            curr_filters = filters * channel_multipliers[i]
            for _ in range(num_res_blocks):
                block_layers.append(ResBlock(prev_filters, curr_filters, **block_args))
                prev_filters = curr_filters
            if i < num_blocks - 1:
                if conv_downsample:
                    block_layers.append(conv_fn(curr_filters, curr_filters, kernel_size=4, stride=2))
                else:
                    block_layers.append(layers.Downsample())
            self.res_blocks.append(block_layers)

        block_layers = nn.ModuleList()
        for _ in range(num_res_blocks):
            block_layers.append(ResBlock(curr_filters, curr_filters, **block_args))
        self.res_blocks.append(block_layers)

        self.norm_out = layers.get_norm_layer(curr_filters, norm_type)()
        self.conv_out = conv_fn(curr_filters, embedding_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.res_blocks:
            for layer in block:
                x = layer(x)
        x = self.norm_out(x)
        x = self.activation_fn(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    """Decoder Blocks."""
    def __init__(
        self,
        config: ml_collections.ConfigDict,
        output_dim: int = 3
    ) -> None:
        super().__init__()

        filters = config.vqvae.filters
        num_res_blocks = config.vqvae.num_res_blocks
        channel_multipliers = config.vqvae.channel_multipliers
        embedding_dim = config.vqvae.embedding_dim
        norm_type = config.vqvae.norm_type
        if config.vqvae.activation_fn == "relu":
            self.activation_fn = F.relu
        elif config.vqvae.activation_fn == "swish":
            self.activation_fn = lambda x: x * torch.sigmoid(x)
        else:
            raise NotImplementedError

        conv_fn = nn.Conv2d
        block_args = dict(
            norm_type=norm_type,
            conv_fn=conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )

        curr_filters = filters * channel_multipliers[-1]
        self.conv_in = conv_fn(embedding_dim, curr_filters, kernel_size=3, padding=1, bias=True)

        num_blocks = len(channel_multipliers)
        self.res_blocks = nn.ModuleList()
        block_layers = nn.ModuleList()
        for _ in range(num_res_blocks):
            block_layers.append(ResBlock(curr_filters, curr_filters, **block_args))
        self.res_blocks.append(block_layers)
        prev_filters = curr_filters
        for i in reversed(range(num_blocks)):
            block_layers = nn.ModuleList()
            curr_filters = filters * channel_multipliers[i]
            for _ in range(num_res_blocks):
                block_layers.append(ResBlock(prev_filters, curr_filters, **block_args))
                prev_filters = curr_filters
            if i > 0:
                block_layers.append(layers.Upsample(2))
                block_layers.append(conv_fn(curr_filters, curr_filters, kernel_size=3, padding=1, bias=True))
            self.res_blocks.append(block_layers)

        self.norm_out = layers.get_norm_layer(curr_filters, norm_type)()
        self.conv_out = conv_fn(curr_filters, output_dim, kernel_size=3, padding=1, bias=True)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.res_blocks:
            for layer in block:
                x = layer(x)
        x = self.norm_out(x)
        x = self.activation_fn(x)
        x = self.conv_out(x)
        return x


class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""
    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = self.config.vqvae.embedding_dim
        codebook_size = self.config.vqvae.codebook_size

        # The JAX version is initialized with variance_scaling(scale=1.0, mode="fan_in", distribution="uniform")
        self.codebook = nn.Embedding(codebook_size, self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z = x.permute(0, 2, 3, 1).contiguous()
        distances = losses.squared_euclidean_distance(z.view(-1, self.embedding_dim), self.codebook.weight)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.codebook(encoding_indices).view(z.shape)

        result_dict = dict()
        if self.training:
            e_latent_loss = torch.mean((quantized.detach()-z)**2) * self.config.vqvae.commitment_cost
            q_latent_loss = torch.mean((quantized - z.detach())**2)
            entropy_loss = 0.0
            if self.config.vqvae.entropy_loss_ratio != 0:
                entropy_loss = losses.entropy_loss(
                    -distances,
                    loss_type=self.config.vqvae.entropy_loss_type,
                    temperature=self.config.vqvae.entropy_temperature
                ) * self.config.vqvae.entropy_loss_ratio
            loss = e_latent_loss + q_latent_loss + entropy_loss
            result_dict = dict(
                quantizer_loss=loss,
                e_latent_loss=e_latent_loss,
                q_latent_loss=q_latent_loss,
                entropy_loss=entropy_loss)
            quantized = z + (quantized - z).detach()

        result_dict.update({
            "encoding_indices": encoding_indices,
            "raw": x,
        })
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, result_dict

    def get_codebook(self) -> torch.Tensor:
        return self.codebook.weight

    def decode_ids(
        self,
        ids: torch.Tensor
    ) -> torch.Tensor:
        feats = self.codebook(ids)
        feats = rearrange(feats, 'b h w c -> b c h w')
        return feats


class VQVAE(nn.Module):
    """VQVAE model."""
    def __init__(
        self,
        config: ml_collections.ConfigDict
    ) -> None:
        super().__init__()
        self.config = config
        if self.config.vqvae.quantizer == "gumbel":
            raise NotImplementedError
        elif self.config.vqvae.quantizer == "vq":
            self.quantizer = VectorQuantizer(config=self.config)
        else:
            raise NotImplementedError
        output_dim = 3
        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config, output_dim=output_dim)

    def encode(
        self,
        input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = input_dict["image"]
        encoded_feature = self.encoder(image)
        if self.config.vqvae.quantizer == "gumbel" and self.training:
            raise NotImplementedError
        else:
            quantized, result_dict = self.quantizer(encoded_feature)
        return quantized, result_dict

    def decode(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        reconstructed = self.decoder(x)
        return reconstructed

    def get_codebook_funct(self):
        return self.quantizer.get_codebook()

    def decode_from_indices(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(inputs, dict):
            ids = inputs["encoding_indices"]
        else:
            ids = inputs
        features = self.quantizer.decode_ids(ids)
        reconstructed_image = self.decode(features)
        return reconstructed_image

    def encode_to_indices(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(inputs, dict):
            image = inputs["image"]
        else:
            image = inputs
        encoded_feature = self.encoder(image)
        _, result_dict = self.quantizer(encoded_feature)
        ids = result_dict["encoding_indices"]
        return ids

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        quantized, result_dict = self.encode(input_dict)
        outputs = self.decoder(quantized)
        return outputs
