import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from collections import deque

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """
    A simplified and robust data loader using a Python generator and tf.data.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    num_devices = jax.device_count()
    if B % num_devices != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by the number of devices ({num_devices})")
    
    device_batch_size = B // num_devices
    
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    def sequence_generator():
        """
        This generator yields token sequences of length T+1.
        """
        token_buffer = deque()

        # This inner generator yields batches of tokenized text
        def _token_batch_generator():
            # Each process reads a different slice of the data
            text_batches = parquets_iter_batched(
                split=split, 
                start=jax.process_index(), 
                step=jax.process_count()
            )
            for text_batch in text_batches:
                token_lists = tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
                yield token_lists

        token_batch_iter = _token_batch_generator()

        while True:
            # Fill the buffer with enough tokens to extract at least one sequence
            while len(token_buffer) < T + 1:
                try:
                    token_lists = next(token_batch_iter)
                    for tokens in token_lists:
                        token_buffer.extend(tokens)
                except StopIteration:
                    # This should not happen with an infinite dataset, but as a safeguard:
                    if len(token_buffer) < T + 1:
                        return # Not enough tokens left to form a full sequence
                    break
            
            # Yield as many full sequences as possible from the buffer
            while len(token_buffer) >= T + 1:
                sequence = [token_buffer.popleft() for _ in range(T + 1)]
                yield np.array(sequence, dtype=np.int32)

    # Create the tf.data.Dataset from our sequence generator
    dataset = tf.data.Dataset.from_generator(
        sequence_generator,
        output_signature=tf.TensorSpec(shape=(T + 1,), dtype=tf.int32)
    )

    # Create inputs and targets (x, y)
    def create_inputs_targets(sequence):
        inputs = sequence[:-1]
        targets = sequence[1:]
        return inputs, targets

    dataset = dataset.map(create_inputs_targets, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch into the final global batch size
    dataset = dataset.batch(B, drop_remainder=True)

    # Reshape for per-device sharding
    def shard_for_devices(inputs, targets):
        inputs = tf.reshape(inputs, (num_devices, device_batch_size, T))
        targets = tf.reshape(targets, (num_devices, device_batch_size, T))
        return inputs, targets

    dataset = dataset.map(shard_for_devices, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch to overlap data loading with training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset.as_numpy_iterator()