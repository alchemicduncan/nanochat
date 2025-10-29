# Gemini Experimental Guide

This document provides guidance for extending the NanoChat repository. Whether you're interested in converting the project to a new framework, scaling up the model, or experimenting with different architectures, this guide provides a starting point for your endeavors.

## Getting Started

Before you begin, ensure you have a solid understanding of the existing codebase. The following files are crucial to understanding the core components of NanoChat:

*   `nanochat/gpt.py`: The core GPT model implementation.
*   `nanochat/engine.py`: The training and evaluation engine.
*   `nanochat/dataloader.py`: The data loading and processing pipeline.
*   `scripts/base_train.py`: The base training script.

### Environment Setup

1.  **Fork the repository:** Create your own fork of the NanoChat repository to store your experiments.
2.  **Create a new branch:** For each new experiment, create a new branch to keep your changes isolated.
    ```bash
    git checkout -b my-experiment
    ```
3.  **Install dependencies:** Follow the instructions in the `README.md` to set up the base environment. You may need to install additional dependencies depending on your experiment.

## Running Experiments

We recommend a structured approach to running experiments to ensure reproducibility and clear tracking of results.

1.  **Configuration:** Use the configuration files in `nanochat/configurator.py` to manage your experiment parameters. Create new configurations for your experiments to avoid modifying the base configurations.
2.  **Logging:** Use a logging framework like Weights & Biases or TensorBoard to track your experiment results. This will help you visualize your results and compare different experiments.
3.  **Code Organization:** Keep your experiment-specific code in separate files or directories to avoid cluttering the main codebase.

## Areas for Exploration

Here are some ideas for extending the NanoChat repository:

### JAX Conversion

Converting the project to JAX can provide significant performance improvements, especially on TPUs.

*   **Key files to modify:** `nanochat/gpt.py`, `nanochat/engine.py`, `nanochat/adamw.py`.
*   **Considerations:** You will need to replace PyTorch tensors and operations with their JAX equivalents. Pay close attention to JAX's functional programming paradigm.

### Running on TPUs

To run NanoChat on TPUs, you will need to make several modifications to the codebase.

*   **Key files to modify:** `nanochat/engine.py`, `nanochat/dataloader.py`.
*   **Considerations:** You will need to use a TPU-compatible data loader and modify the training loop to work with TPUs. A JAX conversion is highly recommended for this.

### Scaling

Scaling up the model size or dataset size can lead to improved performance.

*   **Key files to modify:** `nano.py`, `nanochat/configurator.py`.
*   **Considerations:** You will need to adjust the model and training parameters to handle the increased scale. You may also need to implement more advanced training techniques like distributed training.

### Model Architecture Experiments

Experimenting with different model architectures can lead to new discoveries and improved performance.

*   **Key files to modify:** `nanochat/gpt.py`.
*   **Considerations:** You can experiment with different attention mechanisms, layer normalizations, and activation functions. Be sure to keep the core GPT architecture intact to ensure compatibility with the rest of the codebase.

## Contributing

Welcome contributions to the NanoChat repository. If you have an experiment that you would like to share, please open a pull request with a clear description of your changes and results.

## Project Setup and Workflow

This section provides an overview of the project's structure and the end-to-end workflow, primarily orchestrated by the `speedrun.sh` script. Understanding this process is key to experimenting with and extending NanoChat.

### Environment Setup

The project uses `uv` for Python environment management. The setup process is handled within `speedrun.sh`:

1.  **`uv` installation:** The script checks for `uv` and installs it if not present.
2.  **Virtual Environment:** It creates a local `.venv` if one doesn't exist.
3.  **Dependencies:** It installs dependencies listed in `pyproject.toml` using `uv sync`.
4.  **Activation:** The script sources `.venv/bin/activate` to use the project's virtual environment.

### End-to-End Workflow via `speedrun.sh`

The `speedrun.sh` script automates the entire process of building a NanoChat model from scratch. Here are the key stages:

1.  **Initialization:**
    *   Sets up the base directory for artifacts (`~/.cache/nanochat`).
    *   Resets the reporting module (`nanochat.report reset`).

2.  **Tokenizer Training:**
    *   Installs Rust/Cargo for the `rustbpe` tokenizer.
    *   Builds the tokenizer using `maturin`.
    *   Downloads an initial set of data shards using `nanochat.dataset`.
    *   Trains a BPE tokenizer on the data with `scripts.tok_train`.
    *   Evaluates the tokenizer's compression rate with `scripts.tok_eval`.

3.  **Base Model Pretraining:**
    *   Downloads the `eval_bundle` for CORE metric evaluation.
    *   Downloads the full pretraining dataset (the script calculates the required number of shards based on Chinchilla scaling laws).
    *   Trains the base GPT model using `torchrun` with `scripts.base_train`.
    *   Evaluates the base model's loss (`scripts.base_loss`) and CORE score (`scripts.base_eval`).

4.  **Midtraining:**
    *   Downloads synthetic conversation data.
    *   Conducts midtraining to teach the model conversational structure, tool use, and other capabilities using `scripts.mid_train`.
    *   Evaluates the model on chat-related tasks with `scripts.chat_eval`.

5.  **Supervised Finetuning (SFT):**
    *   Performs SFT to align the model with specific conversational formats using `scripts.chat_sft`.
    *   Re-evaluates the model with `scripts.chat_eval` to measure improvements.

6.  **Reinforcement Learning (RL - Optional):**
    *   The script includes an optional, commented-out step for RL training (`scripts.chat_rl`), currently focused on the GSM8K task.

7.  **Inference and Interaction:**
    *   After training, you can interact with the model via:
        *   **CLI:** `python -m scripts.chat_cli`
        *   **Web UI:** `python -m scripts.chat_web`

8.  **Reporting:**
    *   Finally, `python -m nanochat.report generate` compiles all the evaluation results into a single `report.md` file.

This entire pipeline is designed to be a single, cohesive run, making it easy to replicate results and experiment with changes by modifying the `speedrun.sh` script or the underlying Python scripts.

Happy experimenting!
