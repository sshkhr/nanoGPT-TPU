"""
Sample from a trained model - NNX Version
"""
import os
import pickle
import tiktoken
import jax
import jax.numpy as jnp
from flax import nnx
from model import GPT
from utils import print_compiling

# -----------------------------------------------------------------------------
start = "According to the Buddha, \n"  # or "<|endoftext|>" or whatever you like
num_samples = 2  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
model_type = 'gpt2'  # 'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl'
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

# model - NNX version returns just the model (stateful), not (model, params)
rngs = nnx.Rngs(seed)
override_args = dict(dropout=0.0)
model = GPT.from_pretrained(model_type, rngs, override_args)

model.eval()

# load tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = jnp.array(start_ids, dtype=jnp.int32)[None]
key = jax.random.PRNGKey(seed)

@jax.jit
@print_compiling
def _sample(key, tokens) -> jax.Array:
    # NNX version: model is stateful, no need to pass params separately
    return model.generate(key, tokens, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)


def sample(key, tokens) -> str:
    tokens = _sample(key, tokens)
    return decode(tokens[0])


# run generation
for k in range(num_samples):
    step_key = jax.random.fold_in(key, k)
    sample_str = sample(step_key, x)
    print(sample_str)
    print('---------------')