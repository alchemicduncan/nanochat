import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0 # Import print0

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

    def token_generator():
        print0("Starting token_generator...")
        # This generator will be wrapped by tf.data.Dataset
        
        # infinite iterator over document batches
        while True:
            # Each process will read a different slice of the data
            start_index = jax.process_index()
            step_size = jax.process_count()
            print0(f"Process {jax.process_index()}: Starting to iterate over parquet batches.")
            for batch in parquets_iter_batched(split=split, start=start_index, step=step_size):
                print0(f"Process {jax.process_index()}: Fetched a parquet batch of size {len(batch)}.")
                # Tokenize the documents in parallel
                token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)
                print0(f"Process {jax.process_index()}: Tokenized {len(token_lists)} documents.")
                for tokens in token_lists:
                    for token in tokens:
                        yield token

    # Create a tf.data.Dataset from the generator
    print0("Creating tf.data.Dataset from generator...")
    dataset = tf.data.Dataset.from_generator(
        token_generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    print0("Dataset created.")

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

    # Shard the data across devices
    def shard_data(inputs, targets):
        inputs = tf.reshape(inputs, (num_devices, device_batch_size, T))
        targets = tf.reshape(targets, (num_devices, device_batch_size, T))
        return inputs, targets

    print0("Sharding data across devices...")
    dataset = dataset.map(shard_data, num_parallel_calls=tf.data.AUTOTUNE)
    print0("Data sharded.")

    # Prefetch to overlap data loading with training
    print0("Prefetching data...")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print0("Data prefetching enabled.")

    # Return an iterator that yields JAX arrays
    print0("Returning numpy iterator.")
    return dataset.as_numpy_iterator()