# LLM Fine-tuning & Alignment

Comprehensive implementation of LLM alignment techniques — from supervised fine-tuning to advanced preference optimization methods (RLHF, DPO, ORPO).

## Overview

This project explores different strategies for aligning language models, comparing their effectiveness on the PIQA (Physical Intuition QA) benchmark:

- **Supervised Fine-Tuning (SFT)** — Instruction tuning on labeled data
- **Reward Model Training** — Learning human preferences
- **RLHF (PPO)** — Reinforcement Learning from Human Feedback
- **DPO** — Direct Preference Optimization
- **ORPO** — Odds Ratio Preference Optimization

## Dataset: PIQA

The [PIQA dataset](https://huggingface.co/datasets/ybisk/piqa) tests physical commonsense reasoning:

- Each example contains a **goal** and two **solutions**
- One solution is correct based on common sense
- Models are evaluated by comparing cross-entropy of generated probabilities

Example:
```
Goal: "How to keep a door open"
Solution 1: "Put a doorstop under it" ✓
Solution 2: "Put water under it"
```

## Training Methods

### 1. Supervised Fine-Tuning (SFT)

Fine-tune a base model on prompt-completion pairs:

```python
# Data format
prompt: "Question: {goal}\nAnswer: "
completion: "{correct_solution}"
```

Uses `SFTTrainer` with `DataCollatorForCompletionOnlyLM` to only compute loss on the completion portion.

### 2. Reward Model Training

Train a model to score responses based on human preferences:

```python
# Data format
chosen: "Question: {goal}\nAnswer: {correct_solution}"
rejected: "Question: {goal}\nAnswer: {wrong_solution}"
```

Uses `RewardTrainer` with sequence classification head (single scalar output).

### 3. RLHF with PPO

Full reinforcement learning pipeline:

1. **Policy Model**: Generates responses
2. **Reward Model**: Scores generated responses  
3. **Value Model**: Estimates expected rewards
4. **Reference Model**: KL divergence regularization

```python
# PPO training loop
responses = policy.generate(prompts)
rewards = reward_model(prompts + responses)
stats = ppo_trainer.step(queries, responses, rewards)
```

### 4. Direct Preference Optimization (DPO)

Simplified preference learning without separate reward model:

```python
# Data format for DPOTrainer
{
    'prompt': "Question: {goal}",
    'chosen': "Answer: {correct_solution}",
    'rejected': "Answer: {wrong_solution}"
}
```

DPO directly optimizes the policy to prefer chosen over rejected responses.

### 5. ORPO (Odds Ratio Preference Optimization)

Similar to DPO but uses odds ratio for preference modeling — no reference model needed.

## Evaluation

Uses the [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized evaluation:

```bash
lm_eval \
    --model hf \
    --model_args pretrained={model_id} \
    --tasks piqa \
    --output_path results.json
```

## Project Structure

```
src/
├── data/
│   ├── data_connection.py     # Dataset loading from HuggingFace
│   └── data_processing.py     # Format data for each training method
├── model/
│   └── model_connection.py    # Load models (LM, Reward, PPO)
├── training/
│   ├── base_train.py          # Base trainer class
│   ├── train_supervised.py    # SFT with SFTTrainer
│   ├── train_rlhf.py          # Reward model + PPO training
│   ├── train_dpo.py           # DPO training
│   └── train_orpo.py          # ORPO training
├── evaluation/
│   └── evaluate.py            # Evaluation harness wrapper
├── training_application.py    # Main training orchestration
└── eval_application.py        # Run evaluations
```

## Usage

### Training

```bash
# Supervised fine-tuning
python training_application.py --training_type supervised

# Reward model
python training_application.py --training_type reward

# PPO (RLHF)
python training_application.py --training_type ppo

# DPO
python training_application.py --training_type dpo

# ORPO
python training_application.py --training_type orpo
```

### Evaluation

```bash
python eval_application.py --training_type base       # Baseline
python eval_application.py --training_type supervised # After SFT
python eval_application.py --training_type dpo        # After DPO
```

### AWS SageMaker Training

```bash
python train_sagemaker.py
```

## Key Libraries

- **TRL** — Transformer Reinforcement Learning (SFTTrainer, DPOTrainer, PPOTrainer, etc.)
- **PEFT** — Parameter-Efficient Fine-Tuning
- **Transformers** — HuggingFace model loading
- **Datasets** — Data loading and processing
- **lm-evaluation-harness** — Standardized LLM evaluation

## Data Processing

Each training method requires specific data formats:

| Method | Required Columns |
|--------|-----------------|
| SFT | `text` (prompt + completion) |
| Reward | `input_ids_chosen`, `input_ids_rejected`, `attention_mask_*` |
| PPO | `input_ids`, `attention_mask` (prompt only) |
| DPO | `prompt`, `chosen`, `rejected` |
| ORPO | `prompt`, `chosen`, `rejected` |

## References

- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- [TRL Documentation](https://huggingface.co/docs/trl)

