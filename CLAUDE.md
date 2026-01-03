# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Instruction-Tuned Aspect-Based Sentiment Analysis (ABSA)** research project. It implements fine-tuning of T5 models using instruction-based prompts for aspect extraction and sentiment classification tasks on review text data.

## Development Setup

```bash
conda create -n envname python=3.9 -y
pip install -r requirements.txt
```

Run notebooks from the task-specific directories (ATE_notebook/, ATSC_notebook/, ASPE_notebook/) after updating dataset paths.

## Architecture

### Core Modules (`Imports/`)

- **config.py**: `Config` class with argparse for hyperparameters (learning_rate=5e-5, batch_size=16, epochs=4, weight_decay=0.01)
- **data_prep.py**: `DatasetLoader` class for loading/formatting data into task-specific formats
- **utils.py**: `T5Generator` (Seq2SeqTrainer) and `T5Classifier` (standard Trainer) wrappers around HuggingFace transformers
- **instructions.py**: `InstructionsHandler` with instruction templates (BOS/delimiter/EOS) for each task type

### Task Types

| Task | Description | Output Format |
|------|-------------|---------------|
| **ATE** | Aspect Term Extraction | `term1, term2, ...` |
| **ATSC** | Aspect Term Sentiment Classification | `positive/negative/neutral` |
| **ASPE** | Aspect Sentiment Polarity Extraction | `term1:polarity, term2:polarity, ...` |
| **AOOE** | Aspect Opinion Opinion Extraction | Opinion words for given aspects |
| **AOPE** | Aspect Opinion Pair Extraction | `aspect:opinion, ...` |
| **AOSTE** | Aspect Opinion Sentiment Triple Extraction | `aspect:opinion:sentiment, ...` |

### Instruction Variants

- **InstructABSA-1**: Basic task definitions with example
- **InstructABSA-2**: Enhanced with positive/negative/neutral examples
- **Domain-specific**: `bos_instruct1` (laptops), `bos_instruct2` (restaurants)

### Datasets (`Dataset/`)

SemEval datasets (2014-2016) in CSV format with columns:
- `sentenceId`, `raw_text`, `aspectTerms` (JSON-like string of `[{term, polarity}, ...]`)

## Key Patterns

### Data Processing Flow
1. Load CSV with pandas
2. `DatasetLoader.reconstruct_strings()` to parse nested dict strings
3. `create_data_in_*_format()` to format for specific task (adds BOS/EOS instructions)
4. Convert to HuggingFace Dataset via `set_data_for_training_semeval()`

### Model Training Flow
1. Initialize `T5Generator(model_checkpoint)` (typically `google/flan-t5-base`)
2. Tokenize with `tokenize_function_inputs()` (max_length=512 input, 64 output)
3. Train with `Seq2SeqTrainer`
4. Generate predictions with `get_labels()`
5. Evaluate with `get_metrics()` (Precision, Recall, F1)

### Notebook Naming Convention
`[Task]_Training_&_Inference_[dataset][version].ipynb`
- Tasks: ATE, ATSC, JointTask
- Datasets: lap14, res14, res15, res16
- Versions: (base), v2, v3

## Device Handling

Automatic device selection: CUDA > MPS (Apple Silicon) > CPU
```python
device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')
```
