# Finding Examples Difficult to Translate using GRPO

## Overview

In this project we introduce **MT-GRPO**, a reinforcement learning–based approach for training large language models to generate examples that are difficult to translate. 

Starting form recent small Large Language Models, we modify them using Group Relative Policy Optimization (GRPO) to improve how these models, given a sentence easy to translate, generate examples that are difficult to translate.

Our results demonstrate that MT-GRPO produces high-quality examples that are more challenging for machine translation systems. Our approach outperforms both the base model and alternative methods, demonstrating the effectiveness of the proposed approach.


![Model Pipeline](/assets/main-figure.png)

## Repository Structure

- **`dataset/`**: Contains scripts for loading and processing the datasets (Tatoeba, WMT25, WMT19).
- **`model/`**: Contains the scoring and specific model implementations.
  - `sentinel.py`: Implements the Sentinel metric for difficulty scoring.
  - `comet.py`: Implements COMET scoring.
- **`ext/`**: Directory containing the code for *Sentinel*, the model for phrase hardness scoring.
- **`train.py`**: The main training script using `GRPOTrainer` class.
- **`evaluation_model.py`**: A comprehensive script for evaluating the model against various metrics and judges.
- **`download_data.sh`**: Script to download the required datasets.
- **`download_models.sh`**: Script to download the required translation models.
- **`run_train.sh`**: Script to run the training script on a SLURM cluster.

## Environment Setup

The necessary environment and libraries can be set up using either `venv` or `conda`.

### Option 1: Using venv

Create and activate virtual environment:

```bash
python3 -m venv deep_learning
source deep_learning/bin/activate
pip install --upgrade pip
```

### Option 2: Using Conda

Create and activate conda environment:
```bash
conda create -n deep_learning python=3.11 -y
conda activate deep_learning
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Resources

### 1. Data Download
The project requires WMT25 and Tatoeba datasets; they can be downloaded via:

```bash
./download_data.sh
```

### 2. Model Download
Various translation models are used for evaluating the generated phrases. They are available as Hugging Face models that can be downloaded using:

```bash
./download_models.sh
```
> [!NOTE]
> This requires the `huggingface-cli`. The script will attempt to install `huggingface_hub` if it's missing.

## Usage

### Training

Training the model using GRPO can be performed either by running the python file directly:
   
```bash
python train.py
```

or using the submission script for SLURM clusters:
```bash
sbatch run_train.sh
```

### Evaluation

For evaluating the trained model across datasets (WMT25 and WMT19) using multiple metrics (Sentinel, COMET, Grammar, etc.):

Run:
```bash
python evaluation_model.py
```

## Model Checkpoints

**MT-GRPO** Models are available in a [public polybox folder](https://polybox.ethz.ch/index.php/s/Zx7YNW2RPoMKJtk):

- [llama-8B-sentinel](https://polybox.ethz.ch/index.php/s/syzEiYDJtt7Pxyy): [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) trained via GRPO using Sentinel, grammar correctness and translation quality as rewards.
- [qwen-4B-sentinel](https://polybox.ethz.ch/index.php/s/eFioWRGB8oW4fkj): [Qwen-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) trained via GRPO using Sentinel, grammar correctness and translation quality as rewards.
- [qwen-1.7B-comet](https://polybox.ethz.ch/index.php/s/qGptR8RTjEiZc29): [Qwen-1.7B-Instruct](https://huggingface.co/Qwen/Qwen3-1.7B) trained via GRPO using COMET, grammar correctness and translation quality as rewards.

## Requirements

- *Java 8* or higher is required for `language_tool_python` (for grammatical error checking reward).