from collections import deque
import jax
import jax.numpy as jnp
import numpy as np

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """Stream pretraining text from parquet files, tokenize, yield training batches as JAX arrays."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    num_devices = jax.device_count()
    if B % num_devices != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by the number of devices ({num_devices})")
    
    device_batch_size = B // num_devices
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    # token_buffer holds the tokens for one iteration
    token_buffer = deque()

    # infinite iterator over document batches
    def document_batches():
        while True:
            # Each process will read a different slice of the data
            start_index = jax.process_index()
            step_size = jax.process_count()
            for batch in parquets_iter_batched(split=split, start=start_index, step=step_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]

    batches = document_batches()

    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        
        # Move tokens from the deque into a numpy array
        tokens = np.array([token_buffer.popleft() for _ in range(needed_tokens)], dtype=np.int32)
        
        # Create the inputs/targets as 1D numpy arrays
        inputs_np = tokens[:-1]
        targets_np = tokens[1:]
        
        # Reshape to 2D
        inputs_np = inputs_np.reshape(B, T)
        targets_np = targets_np.reshape(B, T)

        # Reshape for sharding across devices
        inputs_sharded = inputs_np.reshape(num_devices, device_batch_size, T)
        targets_sharded = targets_np.reshape(num_devices, device_batch_size, T)
        
        # Yield JAX arrays
        yield jnp.array(inputs_sharded), jnp.array(targets_sharded)
