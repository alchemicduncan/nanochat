"""
Evaluates the base model's loss on the validation set using JAX.
"""

import os
import jax
import jax.numpy as jnp
import optax
import torchax
from flax.training import train_state

# Enable torchax globally for PyTorch-JAX interoperability
torchax.enable_globally()

from nanochat.common import print0, print_banner
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader_jax import tokenizing_distributed_data_loader

print_banner()

# --- Config ---
# Note: These must match the parameters of the model we want to evaluate
depth = 10
max_seq_len = 1024
device_batch_size = 1 # Per-device batch size
eval_tokens = 20 * 524288 # Number of tokens to evaluate val loss on

# --- JAX/Distributed Setup ---
if jax.process_count() > 1:
    jax.distributed.initialize()
master_process = jax.process_index() == 0
world_size = jax.device_count()
print0(f"JAX process index: {jax.process_index()}, device count: {world_size}")

# --- Model Initialization ---
# NOTE: In a real pipeline, we would load model weights from a checkpoint.
# For now, we re-initialize the model architecture.
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
pt_model = pt_model.to(dtype=jnp.bfloat16)

print0("Wrapping model with torchax and extracting JAX parameters...")
model = pt_model.to('jax')
params, apply_fn = torchax.extract_jax(model)
print0("Model loaded and parameters extracted successfully.")

# --- JAX Evaluation Step ---
def eval_step(params, batch):
    logits = apply_fn(params, batch['inputs'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.reshape(-1, logits.shape[-1]),
        labels=batch['targets'].reshape(-1)
    ).mean()
    return loss

p_eval_step = jax.pmap(eval_step, axis_name='batch')

# --- Main Execution ---
def main():
    print0("\n--- Initializing Data Loader for Validation ---")
    global_batch_sequences = device_batch_size * world_size
    val_loader = tokenizing_distributed_data_loader(
        B=global_batch_sequences,
        T=max_seq_len,
        split="val"
    )
    val_iter = iter(val_loader)
    print0("✅ Data loader initialized.")

    print0("\n--- Starting Evaluation ---")
    # Replicate parameters across devices
    replicated_params = jax.device_put_replicated(params, jax.local_devices())
    
    total_loss = 0.0
    num_batches = eval_tokens // (global_batch_sequences * max_seq_len)
    
    for i in range(num_batches):
        x, y = next(val_iter)
        batch = {'inputs': x.copy(), 'targets': y.copy()}
        loss = p_eval_step(replicated_params, batch)
        total_loss += loss.mean().item()
        print0(f"Batch {i+1}/{num_batches}, Loss: {loss.mean().item():.4f}")

    avg_loss = total_loss / num_batches
    print0(f"\n✅ Evaluation finished.")
    print0(f"Average validation loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
