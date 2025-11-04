"""
Train model using JAX. Run as:

python -m scripts.base_train_jax
"""

import os
import time
import jax
import jax.numpy as jnp
import flax
from flax.experimental import nnx
import optax
import wandb

# Enable torchax globally for PyTorch-JAX interoperability
torchax.enable_globally()

from nanochat.common import print0, print_banner, get_base_dir
from nanochat.gpt_nnx import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader_jax import tokenizing_distributed_data_loader
from nanochat.checkpoint_manager_nnx import save_checkpoint

print_banner()

# --- JAX/Distributed Setup ---
if jax.process_count() > 1:
    jax.distributed.initialize()
master_process = jax.process_index() == 0
world_size = jax.device_count()
print0(f"JAX process index: {jax.process_index()}, device count: {world_size}")

# --- Config ---
depth = 20
max_seq_len = 1024
total_batch_size = 524288
num_iterations = 21400 # From previous logs
device_batch_size = 1 # Per-device batch size

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
rngs = nnx.Rngs(0)
model = GPT(model_config, rngs=rngs)
print0("NNX Model initialized successfully.")

# --- Optimizer ---
def create_optimizer_state(model):
    """Creates initial optimizer state."""
    learning_rate = 0.004 # A common default
    tx = optax.adamw(learning_rate=learning_rate)
    graphdef, params = nnx.split(model)
    return tx.init(params), tx

# --- JAX Training Step ---
def train_step(model, optimizer_state, optimizer, batch):
    def loss_fn(model):
        return model(batch['inputs'], targets=batch['targets'])

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    
    # Average loss and gradients across all devices
    loss = jax.lax.pmean(loss, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    graphdef, params = nnx.split(model)
    _, grads_params = nnx.split(grads)
    
    updates, optimizer_state = optimizer.update(grads_params, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    model = nnx.merge(graphdef, params)
    
    return model, optimizer_state, loss

p_train_step = jax.pmap(train_step, axis_name='batch', in_axes=(0, 0, None, 0))

# --- Main Execution ---
def main():
    # wandb logging init
    run_name = os.environ.get("WANDB_RUN", "dummy")
    use_dummy_wandb = run_name == "dummy" or not master_process
    wandb_run = wandb.init(project="nanochat", name=run_name) if not use_dummy_wandb else type("DummyWandb", (object,), {"log": lambda *args, **kwargs: None, "finish": lambda: None})()

    print0("\n--- Initializing Optimizer ---")
    optimizer_state, optimizer = create_optimizer_state(model)
    model = flax.jax_utils.replicate(model)
    optimizer_state = flax.jax_utils.replicate(optimizer_state)
    print0("✅ Optimizer initialized and state replicated successfully.")

    print0("\n--- Initializing Data Loader ---")
    # Calculate the global batch size in terms of sequences for one forward/backward pass
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
        model, optimizer_state, loss = p_train_step(model, optimizer_state, optimizer, batch)
        
        dt = time.time() - t0
        total_training_time += dt

        # Logging
        mean_loss = loss.mean() # loss is already replicated, just take the mean
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
        
        # Checkpointing logic
        if master_process and (step % 1000 == 0 or step == num_iterations - 1):
            checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", f"d{depth}")
            # We need to unreplicate the model and optimizer_state before saving
            unreplicated_model = flax.jax_utils.unreplicate(model)
            unreplicated_optimizer_state = flax.jax_utils.unreplicate(optimizer_state)
            save_checkpoint(checkpoint_dir, step, unreplicated_model, unreplicated_optimizer_state, {"model_config": model_config_kwargs})

    print0("\n✅ Training loop finished.")
    wandb_run.finish()

if __name__ == "__main__":
    main()