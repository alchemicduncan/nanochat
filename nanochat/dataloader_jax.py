import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, **kwargs):
    """
    Builds a robust, distributed tf.data pipeline for JAX, processing one document at a time.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    num_devices = jax.device_count()
    if B % num_devices != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by the number of devices ({num_devices})")
    
    device_batch_size = B // num_devices
    
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # 1. Get the base dataset of individual documents.
    # This dataset is already sharded by process (e.g., TPU worker).
    dataset = parquets_iter_batched(split=split, start=jax.process_index(), step=jax.process_count())

    # 2. Tokenize each document individually and flatten into a stream of tokens.
    def _tokenize_py_function(document_tensor):
        # Tokenize a single document.
        document_str = document_tensor.numpy().decode('utf-8')
        tokens = tokenizer.encode([document_str], prepend=bos_token, num_threads=1)[0]
        return np.array(tokens, dtype=np.int32)

    def tokenize_and_flatten(document_tensor):
        # This function is used with flat_map. It must return a Dataset.
        tokens_array = tf.py_function(
            _tokenize_py_function,
            inp=[document_tensor],
            Tout=tf.int32
        )
        return tf.data.Dataset.from_tensor_slices(tokens_array)

    dataset = dataset.flat_map(tokenize_and_flatten)

    # 3. Batch the stream of tokens into sequences of length T+1.
    dataset = dataset.batch(T + 1, drop_remainder=True)

    # 4. Create (inputs, targets) tuples.
    def create_inputs_targets(sequence):
        inputs = sequence[:-1]
        targets = sequence[1:]
        return inputs, targets

    dataset = dataset.map(create_inputs_targets, num_parallel_calls=tf.data.AUTOTUNE)

    # 5. Batch again to create the final global batch.
    dataset = dataset.batch(B, drop_remainder=True)

    # 6. Reshape for per-device sharding.
    def shard_for_devices(inputs, targets):
        inputs = tf.reshape(inputs, (num_devices, device_batch_size, T))
        targets = tf.reshape(targets, (num_devices, device_batch_size, T))
        return inputs, targets

    dataset = dataset.map(shard_for_devices, num_parallel_calls=tf.data.AUTOTUNE)

    # 7. Prefetch to overlap data loading with training.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset.as_numpy_iterator()