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

"""Fast decoding routines for non-autoregressive generation."""

from typing import Callable

from einops import rearrange
import torch
import torch.nn.functional as F

from maskgit.libml import mask_schedule

# Confidence score for known tokens to avoid masking or repredicting them.
# Here we don't use 1.0 because the upper bounder of the probability can be
# possiblity larger than 1 due to the noise addition.


def mask_by_random_topk(
    mask_len: int,
    probs: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Modifies from jax.random.choice without replacement.

    JAX's original implementation is as below:
        g = -gumbel(key, (n_inputs,)) - jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]
    We adds temperature annealing on top of it, which is:
        g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]

    Args:
        mask_len: the number to mask.
        probs: the probabilities associated with each entry.
        temperature: when temperature = 1.0, it's identical to jax's implementation.
        The larger this value is, the more random the masking is picked.

    Returns:
        A binary masking map [batch_size, seq_len].
    """
    g = torch.distributions.gumbel.Gumbel(0, 1)
    confidence = torch.log(probs) + temperature * g.sample(probs.shape).to(probs.device)
    sorted_confidence = torch.sort(confidence, dim=-1)[0]
    # Obtains cut off threshold given the mask lengths.
    cut_off = torch.gather(sorted_confidence, -1, mask_len)
    # Masks tokens with lower confidence.
    masking = (confidence < cut_off)
    return masking


class State:
    """Holds decoding state data."""
    def __init__(
        self,
        cur_index: int,  # scalar int32: current decoded length index
        cur_seqs: torch.Tensor,  # int32 [batch, seq_len]
        final_seqs: torch.Tensor  # int32 [batch, num_iter, seq_len]
    ) -> None:
        self.cur_index = cur_index
        self.cur_seqs = cur_seqs
        self.final_seqs = final_seqs

def state_init(
    init_indices: torch.Tensor,
    num_iter: int,
    start_iter: int = 0
) -> State:
    """Initializes the decoding state data structure."""
    final_seqs0 = init_indices.unsqueeze(1)
    final_seqs0 = final_seqs0.repeat(1, num_iter, 1)
    return State(
        cur_index=start_iter, cur_seqs=init_indices, final_seqs=final_seqs0)

def decode(
    inputs: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine"
) -> torch.Tensor:
    """Fast decoding for iterative generation.

    Args:
        inputs: int32 array: [batch_size, seq_length] input sequence of masked
        tokens, where the masking tokens is defined by mask_token_id.
        tokens_to_logits: decoder function taking single token slices and cache and
        returning logits and updated cache.
        mask_token_id: int: [Mask] token id.
        num_iter: int: default is 12.
        start_iter: int: default is 0.
        choice_temperature: float: temperature to control the randomness of masking.
        mask_scheduling_method: masking method string. See mask_schedule.py for
        details.

    Returns:
        [batch_size, num_iter, seq_length] output sequence of tokens in all
        iterations.
    """
    inputs = inputs.long()
    unknown_number_in_the_beginning = torch.sum(inputs == mask_token_id, dim=-1)
    # Initializes state
    state = state_init(inputs, num_iter, start_iter=start_iter)

    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(cur_ids)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        b = probs.shape[0]
        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
        sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)
        
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids

        selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_probs = torch.where(unknown_map, selected_probs,
                                    torch.zeros_like(selected_probs) + torch.inf)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                            mask_scheduling_method)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()

        # Adds noise for randomness
        masking = mask_by_random_topk(mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
        # # Masks tokens with lower confidence.
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        state.cur_index += 1
        state.cur_seqs = sampled_ids

    return state.final_seqs
