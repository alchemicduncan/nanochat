#!/bin/bash

# This script is the JAX-based equivalent of speedrun.sh, designed for TPUs.

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the model using the JAX script
# Note: We are using depth=10 and max_seq_len=1024 to fit in memory
python -m scripts.base_train_jax --depth=10 --max_seq_len=1024 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# The following steps are commented out as they require JAX-specific implementations
# that have not yet been created.

# # evaluate the model on a larger chunk of train/val data and draw some samples
# # torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
# # evaluate the model on CORE tasks
# # torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# # -----------------------------------------------------------------------------
# # Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# # download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# # run midtraining and eval the model
# # torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
# # torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# # -----------------------------------------------------------------------------
# # Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# # train sft and re-eval right away (should see a small bump)
# # torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
# # torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
