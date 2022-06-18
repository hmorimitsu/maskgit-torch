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

from typing import Tuple

from einops import rearrange
import numpy as np
from PIL import ImageFilter, Image
import torch

from maskgit.nets import vqgan_tokenizer, bidirectional_transformer
from maskgit.configs import maskgit_class_cond_config
from maskgit.libml import parallel_decode
from maskgit.utils import Bbox

class ImageNet_class_conditional_generator():
    def checkpoint_canonical_path(
        maskgit_or_tokenizer: str,
        image_size: int
    ) -> str:
        return f"./checkpoints/{maskgit_or_tokenizer}_imagenet{image_size}.ckpt"

    def __init__(
        self,
        image_size: int = 256
    ) -> None:
        maskgit_cf = maskgit_class_cond_config.get_config()
        maskgit_cf.image_size = int(image_size)

        # Define tokenizer
        self.tokenizer_model = vqgan_tokenizer.VQVAE(config=maskgit_cf)
        self.tokenizer_model.eval()
        if torch.cuda.is_available():
            self.tokenizer_model = self.tokenizer_model.cuda()

        # Define transformer
        self.transformer_latent_size = maskgit_cf.image_size // maskgit_cf.transformer.patch_size
        self.transformer_codebook_size = maskgit_cf.vqvae.codebook_size + maskgit_cf.num_class + 1
        self.transformer_block_size = self.transformer_latent_size ** 2 + 1

        self.transformer_model = bidirectional_transformer.BidirectionalTransformer(
            num_image_tokens=self.transformer_block_size,
            num_codebook_vectors=self.transformer_codebook_size,
            dim=maskgit_cf.transformer.num_embeds,
            n_layers=maskgit_cf.transformer.num_layers,
            hidden_dim=maskgit_cf.transformer.intermediate_size,
            num_heads=maskgit_cf.transformer.num_heads,
            attention_dropout=maskgit_cf.transformer.dropout_rate,
            hidden_dropout=maskgit_cf.transformer.dropout_rate)
        self.transformer_model.eval()
        if torch.cuda.is_available():
            self.transformer_model = self.transformer_model.cuda()

        self.maskgit_cf = maskgit_cf

        self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        image_size = self.maskgit_cf.image_size

        ckpt = torch.load(ImageNet_class_conditional_generator.checkpoint_canonical_path("tokenizer", image_size))
        self.tokenizer_model.load_state_dict(ckpt['state_dict'])

        ckpt = torch.load(ImageNet_class_conditional_generator.checkpoint_canonical_path("maskgit", image_size))
        self.transformer_model.load_state_dict(ckpt['state_dict'], strict=False)

    def generate_samples(
        self,
        input_tokens: torch.Tensor,
        start_iter: int = 0,
        num_iterations: int = 16
    ) -> torch.Tensor:
        def tokens_to_logits(seq: torch.Tensor) -> torch.Tensor:
            logits = self.transformer_model(seq)
            logits = logits[..., :self.maskgit_cf.vqvae.codebook_size]
            return logits

        output_tokens = parallel_decode.decode(
            input_tokens,
            tokens_to_logits,
            num_iter=num_iterations,
            choice_temperature=self.maskgit_cf.sample_choice_temperature,
            mask_token_id=self.maskgit_cf.transformer.mask_token_id,
            start_iter=start_iter,
            )
        output_tokens = output_tokens[:, -1, 1:].reshape(-1, self.transformer_latent_size, self.transformer_latent_size)
        gen_images = self.tokenizer_model.decode_from_indices({'encoding_indices': output_tokens})

        return gen_images

    def create_input_tokens_normal(
        self,
        label: torch.Tensor
    ) -> torch.Tensor:
        label_tokens = label * torch.ones([self.maskgit_cf.eval_batch_size, 1])
        # Shift the label by codebook_size 
        label_tokens = label_tokens + self.maskgit_cf.vqvae.codebook_size
        # Create blank masked tokens
        blank_tokens = torch.ones([self.maskgit_cf.eval_batch_size, self.transformer_block_size-1])
        masked_tokens = self.maskgit_cf.transformer.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        input_tokens = torch.cat([label_tokens, masked_tokens], dim=-1)
        if torch.cuda.is_available():
            input_tokens = input_tokens.cuda()
        return input_tokens.long()

    def eval_batch_size(self) -> int:
        return self.maskgit_cf.eval_batch_size

    def _create_input_batch(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        return np.repeat(image[None], self.maskgit_cf.eval_batch_size, axis=0).astype(np.float32)

    def create_latent_mask_and_input_tokens_for_image_editing(
        self,
        image: np.ndarray,
        bbox: Bbox,
        target_label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = self._create_input_batch(image)
        imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2))
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        # Encode the images into image tokens
        image_tokens = self.tokenizer_model.encode_to_indices({"image": imgs})
        image_tokens = rearrange(image_tokens.squeeze(-1), '(b h w) -> b h w', b=self.maskgit_cf.eval_batch_size, h=self.maskgit_cf.image_size//16)

        # Create the masked tokens
        latent_mask = torch.zeros((self.maskgit_cf.eval_batch_size, self.maskgit_cf.image_size//16, self.maskgit_cf.image_size//16)).to(imgs.device)
        latent_t = max(0, bbox.top//16-1)
        latent_b = min(self.maskgit_cf.image_size//16, bbox.height//16+bbox.top//16+1)
        latent_l = max(0, bbox.left//16-1)
        latent_r = min(self.maskgit_cf.image_size//16, bbox.left//16+bbox.width//16+1)
        latent_mask[:, latent_t:latent_b, latent_l:latent_r] = 1

        masked_tokens = (1-latent_mask) * image_tokens + self.maskgit_cf.transformer.mask_token_id * latent_mask
        masked_tokens = rearrange(masked_tokens, 'b h w -> b (h w)')

        # Create input tokens based on the category label
        label_tokens = target_label * torch.ones(self.maskgit_cf.eval_batch_size, 1).to(imgs.device)
        # Shift the label tokens by codebook_size 
        label_tokens = label_tokens + self.maskgit_cf.vqvae.codebook_size
        # Concatenate the two as input_tokens
        input_tokens = torch.cat([label_tokens, masked_tokens], dim=-1)
        return (latent_mask, input_tokens.long())

    def composite_outputs(
        self,
        input: np.ndarray,
        latent_mask: np.ndarray,
        outputs: np.ndarray
    ) -> np.ndarray:
        imgs = self._create_input_batch(input)
        composit_mask = Image.fromarray(np.uint8(latent_mask[0] * 255.))
        composit_mask = composit_mask.resize((self.maskgit_cf.image_size, self.maskgit_cf.image_size))
        composit_mask = composit_mask.filter(ImageFilter.GaussianBlur(radius=self.maskgit_cf.image_size//16-1))
        composit_mask = np.float32(composit_mask)[:, :, np.newaxis] / 255.
        return outputs * composit_mask + (1-composit_mask) * imgs