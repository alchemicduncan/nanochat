import jax
import jax.numpy as jnp
import numpy as np
from collections import deque

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """
    A simple, pure Python data loader that yields sharded JAX arrays.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    num_devices = jax.device_count()
    if B % num_devices != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by the number of devices ({num_devices})")
    
    device_batch_size = B // num_devices
    
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    token_buffer = deque()

    # Create an iterator for tokenized text batches
    text_batches = parquets_iter_batched(
        split=split, 
        start=jax.process_index(), 
        step=jax.process_count()
    )

    while True:
        # The number of tokens needed for one full batch
        needed_tokens = B * T + 1

        # Fill the buffer with enough tokens
        while len(token_buffer) < needed_tokens:
            try:
                text_batch = next(text_batches)
                token_lists = tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)
            except StopIteration:
                # Should not happen with infinite dataset, but as a safeguard
                if len(token_buffer) < needed_tokens:
                    # Not enough tokens to form a full batch, so we're done
                    return
                break
        
        # If we still don't have enough tokens, break
        if len(token_buffer) < needed_tokens:
            break

        # Extract the tokens for the current batch
        batch_tokens = [token_buffer.popleft() for _ in range(B * T)]
        
        # The next token is the target for the last token in the sequence
        # We need to handle the targets carefully
        # Let's create x and y for the whole batch first
        
        # To create x and y, we need B*T+1 tokens
        # Let's adjust the logic to pull B*T+1 tokens and then form x and y
        
        # We need to rethink this part. Let's go back to pulling sequences.
        # The logic from the previous simplified loader was better.
        
        # Let's try again. We need B sequences of length T+1
        
        # We need B * (T+1) tokens to form a batch
        needed_for_batch = B * (T + 1)
        
        while len(token_buffer) < needed_for_batch:
            try:
                text_batch = next(text_batches)
                token_lists = tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)
            except StopIteration:
                if len(token_buffer) < needed_for_batch:
                    return
                break
        
        if len(token_buffer) < needed_for_batch:
            break
            
        # Now, create the batch
        x_batch = np.zeros((B, T), dtype=np.int32)
        y_batch = np.zeros((B, T), dtype=np.int32)
        
        for i in range(B):
            sequence = [token_buffer.popleft() for _ in range(T + 1)]
            x_batch[i] = sequence[:-1]
            y_batch[i] = sequence[1:]
            
        # Shard the data for JAX devices
        x_sharded = x_batch.reshape(num_devices, device_batch_size, T)
        y_sharded = y_batch.reshape(num_devices, device_batch_size, T)
        
        yield jnp.array(x_sharded), jnp.array(y_sharded)