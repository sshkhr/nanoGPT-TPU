## Differences from PyTorch

### Modelling Differences

- There is no equivalent to `ModuleDict` in Flax. 
- For weight tying, we use `flax.nnx.Embed.attend` instead of creating a separate `Linear` layer for the final `lm_head` layer: https://github.com/huggingface/transformers/issues/13086 
- Pytorch `split` behaves differently from JAX and Numpy's `split`: https://github.com/pytorch/pytorch/issues/50012, so the arguments to `jax.numpy.split` is 3 in the `CausalSelfAttention` class.
- For pre-trained model weights, we use `from transformers import FlaxGPT2LMHeadModel`


## Differences from Flax Linen 

### Modelling Differences

- State management is now internal to model, so need need to use an external `params` dictionary. 
- `crop_block_size` in `model.py` aligns much closely with the Pytorch approach due to NNX's ease of model surgery.
- No more `deterministic=not train` for dropout, hence no need to pass `train` to the `__call__` method of MLP, Block, GPT 
