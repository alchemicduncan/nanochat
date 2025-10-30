"""
Train model using JAX. Run as:

python base_train_jax.py

or distributed as:

python base_train_jax.py --jax_distributed
"""

import os
import time
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import flax
import optax
import torchax
import wandb

# Enable torchax globally for PyTorch-JAX interoperability
torchax.enable_globally()

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader_jax import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# JAX compute init
if jax.process_count() > 1:
    jax.distributed.initialize()

master_process = jax.process_index() == 0
world_size = jax.device_count()
print0(f"JAX process index: {jax.process_index()}, device count: {world_size}")

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
byte_tokens = jnp.arange(2, 258)
byte_values = byte_tokens - 2
lookup = jnp.full((vocab_size,), -1, dtype=jnp.int32)
lookup = lookup.at[byte_tokens].set(byte_values)
token_bytes = lookup
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
model_config = GPTConfig(**model_config_kwargs)
pt_model = GPT(model_config)
pt_model.init_weights()
# Wrap the PyTorch model with torchax
model = pt_model.to('jax')
num_params = sum(p.numel() for p in pt_model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = pt_model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

from flax.training import train_state

class TrainState(train_state.TrainState):
    # Add any additional state variables here
    pass

# -----------------------------------------------------------------------------
# Initialize the Optimizer
# Learning rate schedule
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=embedding_lr,
    warmup_steps=round(warmup_ratio * num_iterations),
    decay_steps=num_iterations,
    end_value=final_lr_frac * embedding_lr,
)

tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)

# Initialize the DataLoaders for train/val
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
train_loader = tokenizing_distributed_data_loader(total_batch_size, max_seq_len, split="train")
build_val_loader = lambda: tokenizing_distributed_data_loader(total_batch_size, max_seq_len, split="val")

# -----------------------------------------------------------------------------
# JAX training step
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
    state = state.apply_gradients(grads=grads)
    return state, loss, grads

p_train_step = jax.pmap(train_step, axis_name='batch')

# JAX-compatible sampling function
@jax.jit
def jax_sample_next_token(params, apply_fn, input_ids, rng_key, temperature=1.0, top_k=None):
    # input_ids: (num_devices, 1, T)
    # logits: (num_devices, 1, vocab_size)
    logits = apply_fn(params, input_ids)
    logits = logits[:, -1, :]

    if top_k is not None:
        # Get the top_k logits and their indices
        topk_logits, topk_indices = jax.lax.top_k(logits, k=top_k)
        # Create a mask for the top_k elements
        mask = jnp.full(logits.shape, -jnp.inf)
        mask = jax.vmap(lambda m, i, v: m.at[i].set(v))(mask, topk_indices, topk_logits)
        logits = mask

    if temperature == 0.0:
        next_id = jnp.argmax(logits, axis=-1)
    else:
        # Sample from the softmax distribution
        probs = jax.nn.softmax(logits / temperature, axis=-1)
        next_id = jax.random.categorical(rng_key, probs, axis=-1)

    return next_id.reshape(input_ids.shape[0], 1) # (num_devices, 1)

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training

# Create and initialize the training state
key = jax.random.PRNGKey(0)
# Initialize model parameters properly
params, apply_fn = torchax.extract_jax(model)
state = TrainState.create(apply_fn=apply_fn, params=params, tx=tx)

# Replicate the state across devices
state = flax.jax_utils.replicate(state)

train_iter = iter(train_loader)
x, y = next(train_iter) # kick off load of the very first batch of data

# JAX-compatible evaluate_bpb function
@jax.jit
def jax_evaluate_bpb(params, apply_fn, val_batch, token_bytes_val):
    logits = apply_fn(params, val_batch['inputs'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.reshape(-1, logits.shape[-1]),
        labels=val_batch['targets'].reshape(-1)
    )
    # Only consider non-masked tokens for bpb calculation
    valid_tokens = (val_batch['targets'].reshape(-1) != -1).sum()
    total_loss = loss.sum()
    bpb = (total_loss / valid_tokens) / jnp.log(2.0) # bits per byte
    return bpb

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (total_batch_size * jax.device_count())
        val_bpbs = []
        for _ in range(eval_steps):
            val_x, val_y = next(val_loader)
            val_batch = {'inputs': val_x, 'targets': val_y}
            # pmap over the evaluation function
            sharded_bpb = jax.pmap(jax_evaluate_bpb, axis_name='batch')(state.params, state.apply_fn, val_batch, token_bytes)
            val_bpbs.append(sharded_bpb.mean().item()) # average across devices and convert to scalar
        val_bpb = sum(val_bpbs) / len(val_bpbs)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })

    # once in a while: estimate the CORE metric (all ranks participate)
    # results = {}
    # if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
    #     # TODO: Implement JAX-based CORE metric evaluation
    #     pass

    # once in a while: sample from the model (only on master process)
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        sample_rng = jax.random.split(key, jax.local_device_count()) # Split key for pmap
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            input_ids = jnp.array(tokens, dtype=jnp.int32).reshape(1, -1)
            # Replicate input_ids for pmap
            input_ids = flax.jax_utils.replicate(input_ids)

            generated_tokens = []
            for _ in range(16): # max_tokens
                next_id_sharded = jax_sample_next_token(state.params, state.apply_fn, input_ids, sample_rng, temperature=0.0)
                next_id = next_id_sharded[0].item() # Take from first device and convert to scalar
                generated_tokens.append(next_id)
                input_ids = jnp.concatenate([input_ids, flax.jax_utils.replicate(jnp.array([[next_id]], dtype=jnp.int32))], axis=1)
            print0(tokenizer.decode(tokens + generated_tokens))

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", output_dirname)
        # Create a checkpointer
        checkpointer = ocp.StandardCheckpointer()
        # Save the replicated state. Orbax handles unreplicate automatically.
        checkpointer.save(checkpoint_dir, args=ocp.args.StandardSave(state))
        print0(f"✅ Saved model checkpoint to {checkpoint_dir}")

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        batch = {'inputs': x, 'targets': y}
        state, loss, grads = p_train_step(state, batch)
        x, y = next(train_iter) # prefetch the next batch while the GPU is busy with forward/backward

    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    # The loss from p_train_step is sharded, so we need to average it across devices
    mean_loss = jax.lax.pmean(loss, axis_name='batch').mean()
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * mean_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    # tok_per_sec = int(world_tokens_per_fwdbwd / dt) # TODO: JAX equivalent
    # flops_per_sec = num_flops_per_token * total_batch_size / dt # TODO: JAX equivalent
    # promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    # mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in % # TODO: JAX equivalent
    # if step > 10:
    #     total_training_time += dt # only count the time after the first 10 steps
    # print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    # if step % 100 == 0:
    #     wandb_run.log({
    #         "step": step,
    #         "total_training_flops": flops_so_far,
    #         "total_training_time": total_training_time,
    #         "train/loss": debiased_smooth_loss,
    #         "train/lrm": lrm,
    #         "train/dt": dt,
    #         "train/tok_per_sec": tok_per_sec,
    #         "train/mfu": mfu,
    #     })

    if master_process and step % 10 == 0:
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}")
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
        })

# print a few more stats
# print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
# print0(f"Total training time: {total_training_time/60:.2f}m")
# print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
# from nanochat.report import get_report
# get_report().log(section="Base model training", data=[
#     user_config, # CLI args
#     { # stats about the training setup
#         "Number of parameters": num_params,
#         "Number of FLOPs per token": f"{num_flops_per_token:e}",
#         "Calculated number of iterations": num_iterations,
#         "Number of training tokens": total_tokens,
#         "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
#         "DDP world size": ddp_world_size,
#         "warmup_ratio": warmup_ratio,
#         "warmdown_ratio": warmdown_ratio,
#         "final_lr_frac": final_lr_frac,
#     },
#     { # stats about training outcomes
#         "Minimum validation bpb": min_val_bpb,
#         "Final validation bpb": val_bpb,
#         "CORE metric estimate": results.get("core_metric", None),
#         "MFU %": f"{mfu:.2f}%",
#         "Total training flops": f"{flops_so_far:e}",
#         "Total training time": f"{total_training_time/60:.2f}m",
#         "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
#     }
# ])

# cleanup
wandb_run.finish() # wandb run finish
# compute_cleanup() # TODO: JAX equivalent
