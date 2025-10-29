# TPU Conversion Plan

## 1. Objective

The primary goal is to adapt the NanoChat training pipeline to run efficiently on Google Cloud TPUs. This conversion will focus on leveraging JAX for the training loop and data parallelism while keeping the core `nn.Module` model definition in PyTorch. This hybrid approach, utilizing `torchax`, aims to minimize code changes to the model architecture while unlocking the performance benefits of TPUs.

## 2. Technical Approach: PyTorch/JAX Bridge

We will use the `torchax` library to bridge PyTorch and JAX. The strategy is as follows:

1.  **Model Definition:** The `GPT` model in `nanochat/gpt.py`, which is an `nn.Module`, will remain unchanged. We may explore using `nng` for model definition in future iterations, but for this conversion, we will leverage the existing PyTorch model.
2.  **Model Loading:** We will load the PyTorch model and its weights directly into a JAX-compatible format. `torchax` provides utilities for this.
3.  **Training Loop:** The training loops in `scripts/base_train.py`, `scripts/mid_train.py`, `scripts/chat_sft.py`, and `scripts/chat_rl.py` will be rewritten in JAX.
4.  **Data Parallelism:** JAX's `pmap` (parallel map) will be used to distribute the training across multiple TPU cores.

This approach allows us to focus on the "hot path" – the training loop – without needing to translate the entire model architecture to a different framework like Flax.

## 3. Phased Implementation Plan

The conversion will be executed in the following phases:

### Phase 1: Environment and Setup

*   **Dependencies:** Add `jax`, `flax`, `optax`, and `torchax` to `pyproject.toml`.
*   **TPU Environment:** Ensure the development environment is configured to access and utilize TPUs.
*   **New Scripts:** Create new training scripts for the JAX implementation, e.g., `scripts/base_train_jax.py`, to work in parallel with the existing CUDA-based scripts.

### Phase 2: Proof of Concept (`base_train_jax.py`)

*   **Model Loading:** In `scripts/base_train_jax.py`, implement the logic to load the PyTorch `GPT` model and its weights into a JAX-compatible format.
*   **JAX Data Pipeline:** Modify `nanochat/dataloader.py` or create a new JAX-specific data loader to efficiently feed data to TPUs. This will involve sharding the data across TPU devices.
*   **JAX Training Step:**
    *   Define a JAX loss function that takes the model parameters and a batch of data.
    *   Create a JAX training step function that:
        *   Calculates the loss.
        *   Uses `jax.grad` to compute gradients.
        *   Applies gradients using an Optax optimizer (e.g., `optax.adamw`).
    *   Wrap the training step in `pmap` to enable data parallelism.

### Phase 3: JAX Training Loop

*   **Implement the Loop:** In `scripts/base_train_jax.py`, write the main training loop that:
    *   Initializes the model and optimizer states.
    *   Iterates through the JAX data loader.
    *   Executes the `pmapped` training step.
    *   Logs metrics (loss, learning rate, etc.) to `wandb` and the console.
    *   Handles checkpointing of the JAX model state (e.g., using `orbax`).
*   **Configuration:** Adapt `nanochat/configurator.py` to handle JAX-specific hyperparameters.

### Phase 4: Integration and Testing

*   **`speedrun_tpu.sh`:** Create a new script, `speedrun_tpu.sh`, that mirrors `speedrun.sh` but calls the new JAX-based training scripts.
*   **Testing:** Thoroughly test the `base_train_jax.py` script on a TPU to ensure correctness and performance.

### Phase 5: Convert Other Training Scripts

*   Once the `base_train` conversion is successful, apply the same pattern to the other training scripts:
    *   `scripts/mid_train.py` -> `scripts/mid_train_jax.py`
    *   `scripts/chat_sft.py` -> `scripts/chat_sft_jax.py`
    *   `scripts/chat_rl.py` -> `scripts/chat_rl_jax.py`

## 4. File Modifications

### New Files

*   `TPU_CONVERSION_PLAN.md`: This file.
*   `speedrun_tpu.sh`: A new run script for TPU-based training.
*   `scripts/base_train_jax.py`: JAX version of `base_train.py`.
*   `scripts/mid_train_jax.py`: JAX version of `mid_train.py`.
*   `scripts/chat_sft_jax.py`: JAX version of `chat_sft.py`.
*   `scripts/chat_rl_jax.py`: JAX version of `chat_rl.py`.
*   `nanochat/jax_utils.py` (optional): A utility file for common JAX functions (e.g., checkpointing, metric computation).

### Modified Files

*   `pyproject.toml`: To add new JAX and `torchax` dependencies.
*   `nanochat/dataloader.py`: May need modifications to support JAX data sharding, or a new JAX-specific dataloader will be created.
*   `.gitignore`: To exclude JAX-related artifacts.

## 5. Key Challenges

*   **Data Loading:** The current data loader is tightly coupled with PyTorch. Adapting it for efficient TPU data feeding will be a critical task.
*   **Debugging:** Debugging on TPUs can be challenging. We will need to rely on JAX's debugging tools and careful logging.
*   **Optimizer States:** The current implementation uses custom optimizers (`DistAdamW`, `DistMuon`). These will need to be replaced with equivalent optimizers from Optax, and their states will need to be managed in the JAX training loop.
*   **Mixed Precision:** Ensuring that mixed-precision training (`bfloat16`) is correctly implemented in the JAX training loop is crucial for performance.
