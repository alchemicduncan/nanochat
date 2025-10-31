import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from collections import deque
import pyarrow.parquet as pq

from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def _parquets_iter_py(split, start=0, step=1):
    """
    A Python generator that iterates through the documents in the dataset.
    This is a pure python implementation to be used inside tf.data.Dataset.from_generator.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            try:
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for text in texts:
                    yield text
            except Exception as e:
                print(f"Warning: Could not read row group {rg_idx} from {filepath}: {e}")
                continue

def tokenizing_distributed_data_loader(B, T, split, tokenizer_batch_size=128, **kwargs):
    """
    Builds a robust, distributed tf.data pipeline for JAX, processing documents in batches.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    num_devices = jax.device_count()
    device_batch_size = B
    global_batch_size = B * num_devices
    
    def data_generator():
        tokenizer = get_tokenizer()
        bos_token = tokenizer.get_bos_token_id()
        token_buffer = deque()
        
        doc_iterator = _parquets_iter_py(split=split, start=jax.process_index(), step=jax.process_count())

        while True:
            # Fill buffer until we have enough for at least one sequence
            while len(token_buffer) < T + 1:
                try:
                    # For efficiency, tokenize in batches
                    doc_batch = [next(doc_iterator) for _ in range(tokenizer_batch_size)]
                    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)
                    for tokens in token_lists:
                        token_buffer.extend(tokens)
                except StopIteration:
                    # No more documents, break inner loop
                    break
            
            # If after trying to fill, we still don't have enough, we are done.
            if len(token_buffer) < T + 1:
                return # This ends the generator

            # Yield one sequence
            sequence = [token_buffer.popleft() for _ in range(T + 1)]
            inputs = np.array(sequence[:-1], dtype=np.int32)
            targets = np.array(sequence[1:], dtype=np.int32)
            yield inputs, targets

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(T,), dtype=tf.int32),
            tf.TensorSpec(shape=(T,), dtype=tf.int32)
        )
    )

    # Batch the (inputs, targets) tuples and then shard for devices.
    dataset = dataset.batch(global_batch_size, drop_remainder=True)

    def shard_for_devices(inputs, targets):
        inputs = tf.reshape(inputs, (num_devices, device_batch_size, T))
        targets = tf.reshape(targets, (num_devices, device_batch_size, T))
        return inputs, targets

    dataset = dataset.map(shard_for_devices, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset.as_numpy_iterator()