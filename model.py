"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) Karpathy's original NanoGPT model in Pytorch:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core import freeze
from flax import traverse_util
import optax


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class CausalSelfAttention(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, use_bias=config.bias, rngs=rngs)
        # output projection
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, use_bias=config.bias, rngs=rngs)
        # regularization
        self.attn_dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.resid_dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)  # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)  # (B, nh, T, hs)

        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nnx.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.swapaxes(1, 2).reshape(B, T, C)  # re-assemble all head outputs side by side

        # outside projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4*config.n_embd, use_bias=config.bias, rngs=rngs)
        self.c_proj = nnx.Linear(4*config.n_embd, config.n_embd, use_bias=config.bias, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, use_bias=config.bias, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, use_bias=config.bias, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        assert config.vocab_size is not None
        assert config.block_size is not None

        # Note: FLAX does not have the equivalent of PyTorch's ModuleDict
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(config.block_size, config.n_embd,  rngs=rngs)
        self.drop = nnx.Dropout(config.dropout, rngs=rngs)

        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(config.n_embd, use_bias=config.bias, rngs=rngs)

        # Note: Don't need to define output projection separately, can use attend
        # see: https://github.com/google/flax/discussions/2186

    def __call__(self, idx: jax.Array, targets: Optional[jax.Array] = None):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32)[None]  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits = self.wte.attend(x)  # https://paperswithcode.com/method/weight-tying

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets).mean()
        else:
            # Inference: only need logits for the last position
            # logits = self.wte.attend(x[:, [-1], :])  # Only last position
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        """
        model surgery to decrease the block size if necessary
        e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        # Crop position embeddings directly
        original_embeddings = self.wpe.embedding.value
        self.wpe.embedding.value = original_embeddings[:block_size]

    @classmethod
    def from_pretrained(cls, model_type: str, rngs: nnx.Rngs, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import FlaxGPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config, rngs=nnx.Rngs(0))

        # init a huggingface/transformers model
        model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
        hf_params = model_hf.params['transformer']

        # Create NNX model instance
        model = GPT(config, rngs)

        # Convert and load weights
        model.wte.embedding.value = hf_params['wte']['embedding']
        model.wpe.embedding.value = hf_params['wpe']['embedding']

        model.ln_f.scale.value = hf_params['ln_f']['scale']
        if config.bias:
            model.ln_f.bias.value = hf_params['ln_f']['bias']

        for i in range(config.n_layer):
            block_hf = hf_params['h'][str(i)]
            block_nnx = model.h[i]

            block_nnx.ln_1.scale.value = block_hf['ln_1']['scale']
            block_nnx.ln_2.scale.value = block_hf['ln_2']['scale']
            if config.bias:
                block_nnx.ln_1.bias.value = block_hf['ln_1']['bias']
                block_nnx.ln_2.bias.value = block_hf['ln_2']['bias']

            c_attn_kernel = block_hf['attn']['c_attn']['kernel'].T
            block_nnx.attn.c_attn.kernel.value = c_attn_kernel

            block_nnx.attn.c_proj.kernel.value = block_hf['attn']['c_proj']['kernel'].T

            if config.bias:
                block_nnx.attn.c_attn.bias.value = block_hf['attn']['c_attn']['bias']
                block_nnx.attn.c_proj.bias.value = block_hf['attn']['c_proj']['bias']

            # MLP weights
            block_nnx.mlp.c_fc.kernel.value = block_hf['mlp']['c_fc']['kernel'].T
            block_nnx.mlp.c_proj.kernel.value = block_hf['mlp']['c_proj']['kernel'].T

            if config.bias:
                block_nnx.mlp.c_fc.bias.value = block_hf['mlp']['c_fc']['bias']
                block_nnx.mlp.c_proj.bias.value = block_hf['mlp']['c_proj']['bias']

        return model

    def get_num_params(self, non_embedding=True, breakdown=False):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding (bool): If True, exclude position embeddings from count.
                Token embeddings are kept because they're shared with the final layer.
            breakdown (bool): If True, return dict with decay/no-decay breakdown.

        Returns:
            int or dict: Total parameter count, or breakdown by parameter type.
        """
        _, params, _ = nnx.split(self, nnx.Param, ...)

        def is_decay_param(path):
            """Helper function to determine if parameter should have weight decay"""
            param_name = path[-1]
            return param_name in ('kernel', 'embedding')

        total_params = 0
        decay_params = 0
        nodecay_params = 0
        decay_count = 0
        nodecay_count = 0
        pos_embed_params = 0

        def count_param(path, param):
            nonlocal total_params, decay_params, nodecay_params, decay_count, nodecay_count, pos_embed_params
            param_size = param.size
            total_params += param_size

            # Track position embedding parameters (for non_embedding flag)
            if len(path) >= 2 and path[-2] == 'wpe' and path[-1] == 'embedding':
                pos_embed_params += param_size

            # Track decay vs no-decay breakdown
            if is_decay_param(path):
                decay_params += param_size
                decay_count += 1
            else:
                nodecay_params += param_size
                nodecay_count += 1

        from flax.traverse_util import path_aware_map
        path_aware_map(count_param, params)

        # Apply non_embedding filter if requested
        final_total = total_params - pos_embed_params if non_embedding else total_params

        if breakdown:
            return {
                'total': final_total,
                'total_with_pos_embed': total_params,
                'pos_embed': pos_embed_params,
                'decay_params': decay_params,
                'nodecay_params': nodecay_params,
                'decay_count': decay_count,
                'nodecay_count': nodecay_count
            }
        else:
            return final_total

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Configure optimizer - corresponds to PyTorch version but uses different partitioning strategy.

        PyTorch version uses dimension-based partitioning (dim >= 2 vs dim < 2)
        NNX version uses name-based partitioning for clarity and explicitness.

        Both achieve the same result:
        - Weight matrices and embeddings get weight decay
        - Biases and layer norm parameters don't get weight decay
        """

        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate, 
                b1=betas[0], 
                b2=betas[1],
                weight_decay=decay
            )

        def partition_fn(path: Tuple[str, ...], x) -> str:
            """
            Partition function - equivalent to PyTorch's dimension-based approach:

            PyTorch logic:
            - p.dim() >= 2 -> decay (weight matrices, embeddings) 
            - p.dim() < 2 -> no_decay (biases, layer norm scales)

            Our name-based logic (equivalent but more explicit):
            - 'kernel' -> decay (weight matrices - always 2D)
            - 'embedding' -> decay (embedding matrices - always 2D) 
            - 'bias' -> no_decay (bias vectors - always 1D)
            - 'scale' -> no_decay (layer norm scales - always 1D)
            """
            param_name = path[-1]
            if param_name in ('bias', 'scale'):
                return 'no_decay'
            elif param_name in ('kernel', 'embedding'):
                return 'decay'
            else:
                raise ValueError(f"Unrecognized parameter: {path}")

        # Get only the trainable parameters (equivalent to requires_grad filter)
        _, params, _ = nnx.split(self, nnx.Param, ...)

        # Create partition map (equivalent to creating decay_params and nodecay_params lists)
        partition_optimizers = {    
            'decay': get_optimizer(weight_decay), 
            'no_decay': get_optimizer(0.0)
        }

        # Apply partitioning function
        from flax.traverse_util import path_aware_map
        from flax.core import freeze
        param_partitions = freeze(path_aware_map(partition_fn, params))

        # Create multi-transform optimizer (equivalent to PyTorch's parameter groups)
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        # Get parameter breakdown for logging (reuse get_num_params)
        param_info = self.get_num_params(breakdown=True)
        print(f"num decayed parameter tensors: {param_info['decay_count']}, with {param_info['decay_params']:,} parameters")
        print(f"num non-decayed parameter tensors: {param_info['nodecay_count']}, with {param_info['nodecay_params']:,} parameters")

        return tx

    def generate(self, key, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (Int32 Tensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=1)
        indexes = jnp.arange(T, T + max_new_tokens)

        def scan_f(carry, i):
            tokens, key = carry
            step_key, key = jax.random.split(key)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(tokens)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, i - 1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                # instead of calculating softmax and then sampling next index, we directly pass logits to jax.random.categorical
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)

            # Update tokens
            tokens = tokens.at[:, i].set(next_token)
            return (tokens, key), None

        (tokens, _), _ = jax.lax.scan(scan_f, (tokens, key), indexes)
        return tokens
