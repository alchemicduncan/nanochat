import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

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
    dataset = parquets_iter_batched(split=split, start=jax.process_index(), step=jax.process_count())

    # Batch documents for tokenization efficiency
    dataset = dataset.batch(tokenizer_batch_size)

    # Tokenize the documents
    def _tokenize_function(documents_tensor):
        # documents_tensor is a 1D tensor of strings. Decode them for the tokenizer.
        documents_np = [d.decode('utf-8') for d in documents_tensor.numpy()]
        token_lists = tokenizer.encode(documents_np, prepend=bos_token, num_threads=tokenizer_threads)
        # Flatten the list of lists into a single list of tokens
        flat_tokens = [token for sublist in token_lists for token in sublist]
        return np.array(flat_tokens, dtype=np.int32)

    # Apply the tokenization function
    dataset = dataset.map(
        lambda documents: tf.py_function(
            _tokenize_function,
            inp=[documents],
            Tout=tf.int32
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Flatten the dataset of token arrays into a single stream of tokens
    dataset = dataset.flat_map(lambda tokens: tf.data.Dataset.from_tensor_slices(tokens))

    # Batch the tokens into sequences of length T+1
    dataset = dataset.batch(T + 1, drop_remainder=True)

    # Create inputs and targets
    def create_inputs_targets(tokens):
        inputs = tokens[:-1]
        targets = tokens[1:]
        return inputs, targets

    dataset = dataset.map(create_inputs_targets, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch again to create the final batch of size B
    dataset = dataset.batch(B, drop_remainder=True)

    # Reshape for per-device sharding
    def shard_for_devices(inputs, targets):
        inputs = tf.reshape(inputs, (num_devices, device_batch_size, T))
        targets = tf.reshape(targets, (num_devices, device_batch_size, T))
        return inputs, targets

    dataset = dataset.map(shard_for_devices, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch to overlap data loading with training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Return an iterator that yields JAX arrays
    return dataset.as_numpy_iterator()