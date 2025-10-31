"""
Train model using JAX. Run as:

python -m scripts.base_train_jax
"""

import os
import time
import jax
import jax.numpy as jnp
import flax
import optax
import torchax
import wandb
from flax.training import train_state

# Enable torchax globally for PyTorch-JAX interoperability
torchax.enable_globally()

from nanochat.common import print0, print_banner, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader_jax import tokenizing_distributed_data_loader

print_banner()

# --- JAX/Distributed Setup ---
if jax.process_count() > 1:
    jax.distributed.initialize()
master_process = jax.process_index() == 0
print0(f"JAX process index: {jax.process_index()}, device count: {jax.device_count()}")

# --- Config ---
depth = 20
max_seq_len = 2048
total_batch_size = 524288
num_iterations = 21400 # From previous logs
device_batch_size = 32 # Per-device batch size

# --- Model Initialization ---
print0("Initializing model...")
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads

model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim
)
model_config = GPTConfig(**model_config_kwargs)
pt_model = GPT(model_config)
pt_model.init_weights()

print0("Wrapping model with torchax and extracting JAX parameters...")
model = pt_model.to('jax')
params, apply_fn = torchax.extract_jax(model)
print0("Model loaded and parameters extracted successfully.")