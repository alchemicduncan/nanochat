"""
Utilities for saving and loading NNX model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import jax
import jax.numpy as jnp
from flax.experimental import nnx
import optax

from nanochat.common import get_base_dir
from nanochat.gpt_nnx import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model: nnx.Module, optimizer_state, meta_data):
    assert int(os.environ.get('RANK', 0)) == 0 # prevent footguns for now
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the model state (parameters)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.msgpack")
    nnx.save(model, model_path)
    log0(f"Saved model file to: {model_path}")

    # Save the optimizer state
    if optimizer_state is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.msgpack")
        # Optax states are PyTree, so we can save them directly
        with open(optimizer_path, "wb") as f:
            f.write(nnx.msgpack_serialize(optimizer_state))
        log0(f"Saved optimizer file to: {optimizer_path}")

    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")

def load_checkpoint(checkpoint_dir, step, load_optimizer=False):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.msgpack")
    # For loading, we need to reconstruct the model architecture first
    # The actual model parameters will be loaded into this structure
    
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    
    model_config_kwargs = meta_data["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    rngs = nnx.Rngs(0) # Rngs are needed for initialization, but actual params will be loaded
    model = GPT(model_config, rngs=rngs)
    
    # Load the saved parameters into the model
    nnx.load(model_path, model)

    # Load the optimizer state if requested
    optimizer_state = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.msgpack")
        with open(optimizer_path, "rb") as f:
            optimizer_state = nnx.msgpack_restore(f.read())

    return model, optimizer_state, meta_data

def build_model(checkpoint_dir, step, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model, optimizer_state, meta_data = load_checkpoint(checkpoint_dir, step, load_optimizer=False)
    
    # Put the model in the right training phase / mode
    # NNX models don't have explicit train/eval modes like PyTorch
    # This might need adjustment based on how NNX handles this.
    
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == meta_data["model_config"]["vocab_size"]
    return model, tokenizer, meta_data

def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]

def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.msgpack with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.msgpack"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
