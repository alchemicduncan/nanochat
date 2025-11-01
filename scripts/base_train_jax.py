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
world_size = jax.device_count()
print0(f"JAX process index: {jax.process_index()}, device count: {world_size}")

# --- Config ---
depth = 20
max_seq_len = 2048
total_batch_size = 524288
num_iterations = 21400 # From previous logs
device_batch_size = 8 # Per-device batch size

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

# --- Optimizer and TrainState ---
class TrainState(train_state.TrainState):
    # A simple extension of TrainState to hold any additional state we might need
    pass

def create_train_state(params, apply_fn):
    """Creates initial TrainState."""
    learning_rate = 0.004 # A common default
    tx = optax.adamw(learning_rate=learning_rate)
    return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)

# --- JAX Training Step ---
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['inputs'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape(-1, logits.shape[-1]),
            labels=batch['targets'].reshape(-1)
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Add pmean here for cross-device averaging
    loss = jax.lax.pmean(loss, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    state = state.apply_gradients(grads=grads)
    return state, loss

p_train_step = jax.pmap(train_step, axis_name='batch')

# --- Main Execution ---
def main():
    # wandb logging init
    run_name = os.environ.get("WANDB_RUN", "dummy")
    use_dummy_wandb = run_name == "dummy" or not master_process
    wandb_run = wandb.init(project="nanochat", name=run_name) if not use_dummy_wandb else type("DummyWandb", (object,), {"log": lambda *args, **kwargs: None, "finish": lambda: None})()

    print0("\n--- Initializing Optimizer and TrainState ---")
    state = create_train_state(params, apply_fn)
    state = flax.jax_utils.replicate(state)
    print0("✅ Optimizer and TrainState initialized and replicated successfully.")

    print0("\n--- Initializing Data Loader ---")
    # Calculate the global batch size in terms of sequences
    global_batch_sequences = device_batch_size * world_size
    train_loader = tokenizing_distributed_data_loader(
        B=global_batch_sequences,
        T=max_seq_len,
        split="train"
    )
    train_iter = iter(train_loader)
    print0("✅ Data loader initialized.")

    print0("\n--- Starting Training Loop ---")
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    ema_beta = 0.9
    total_training_time = 0.0

    for step in range(num_iterations):
        t0 = time.time()
        x, y = next(train_iter) # Fetch batch
        batch = {'inputs': x.copy(), 'targets': y.copy()} # Ensure writable copies
        state, loss = p_train_step(state, batch)
        dt = time.time() - t0
        total_training_time += dt

        # Logging
        # The loss is already averaged across devices, so we just take the mean of the sharded loss tensor
        mean_loss = loss.mean()
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * mean_loss.item()
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * (step + 1) / num_iterations

        if master_process and (step % 10 == 0 or step == num_iterations - 1):
            print0(f"Step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | dt: {dt * 1000:.2f}ms")
            wandb_run.log({
                "step": step,
                "train/loss": debiased_smooth_loss,
                "total_training_time": total_training_time,
            })
        
        # TODO: Add evaluation and checkpointing logic

    print0("\n✅ Training loop finished.")
    wandb_run.finish()

if __name__ == "__main__":
    main()