import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """Stream pretraining text from parquet files, tokenize, yield training batches as JAX arrays."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    num_devices = jax.device_count()
    if B % num_devices != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by the number of devices ({num_devices})")
    
    device_batch_size = B // num_devices
    
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # Get the base tf.data.Dataset of raw text documents
    print0("Creating base tf.data.Dataset from parquets_iter_batched...")
    dataset = parquets_iter_batched(split=split, start=jax.process_index(), step=jax.process_count())
    print0("Base dataset created.")

    # Tokenize the documents
    def _tokenize_function(documents):
        # documents will be a list of strings (from tf.py_function in dataset.py)
        # We need to convert tf.Tensor of strings to numpy array of strings
        documents_np = [d.numpy().decode('utf-8') for d in documents]
        token_lists = tokenizer.encode(documents_np, prepend=bos_token, num_threads=tokenizer_threads)
        # Flatten the list of lists into a single list of tokens
        flat_tokens = [token for sublist in token_lists for token in sublist]
        return tf.constant(flat_tokens, dtype=tf.int32)

    print0("Applying tokenization to the dataset...")
    # Use flat_map to handle the list of token lists returned by tokenizer.encode
    dataset = dataset.flat_map(
        lambda documents: tf.data.Dataset.from_tensor_slices(
            tf.py_function(
                _tokenize_function,
                inp=[documents],
                Tout=tf.int32
            )
        )
    )
    print0("Tokenization applied.")

    # Batch the tokens into sequences of length T+1
    print0(f"Batching tokens into sequences of length {T + 1}...")
    dataset = dataset.batch(T + 1, drop_remainder=True)
    print0("Tokens batched.")

    # Create inputs and targets
    def create_inputs_targets(tokens):
        inputs = tokens[:-1]
        targets = tokens[1:]
        return inputs, targets

    print0("Mapping dataset to create inputs and targets...")
    dataset = dataset.map(create_inputs_targets, num_parallel_calls=tf.data.AUTOTUNE)
    print0("Inputs and targets created.")

    # Batch again to create the final batch of size B
    print0(f"Batching again to create final batch of size {B}...")
    dataset = dataset.batch(B, drop_remainder=True)
    print0("Final batch created.")

    # Shard the data across devices (this is now handled by the initial `start` and `step` in parquets_iter_batched)
    # The dataset is already sharded per process, now we need to reshape for per-device sharding within a process
    def shard_for_devices(inputs, targets):
        inputs = tf.reshape(inputs, (num_devices, device_batch_size, T))
        targets = tf.reshape(targets, (num_devices, device_batch_size, T))
        return inputs, targets

    print0("Reshaping for per-device sharding...")
    dataset = dataset.map(shard_for_devices, num_parallel_calls=tf.data.AUTOTUNE)
    print0("Data reshaped for devices.")

    # Prefetch to overlap data loading with training
    print0("Prefetching data...")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print0("Data prefetching enabled.")

    # Return an iterator that yields JAX arrays
    print0("Returning numpy iterator.")
    return dataset.as_numpy_iterator()