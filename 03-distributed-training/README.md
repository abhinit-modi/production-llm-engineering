# Distributed Training

Production-ready distributed training infrastructure using HuggingFace Accelerate, DeepSpeed ZeRO optimization, and Weights & Biases experiment tracking.

## Overview

This project implements scalable training pipelines that can run on single GPUs or distributed across multiple machines:

- **Basic Trainer** — Standard PyTorch training loop
- **Accelerated Trainer** — Distributed training with HuggingFace Accelerate
- **ZeRO Optimization** — Memory-efficient distributed training with DeepSpeed
- **W&B Integration** — Experiment tracking and visualization
- **SageMaker Deployment** — Cloud training on AWS

## Training Pipeline

A training application follows these steps:

1. **Data Loading** — Pull data from HuggingFace datasets
2. **Data Processing** — Tokenize and format for training
3. **Model Loading** — Load pretrained model and tokenizer
4. **Training Loop** — Forward pass, backward pass, optimization
5. **Evaluation** — Compute metrics on validation set
6. **Model Saving** — Push to HuggingFace Hub

## Implementations

### Basic Trainer

Standard PyTorch training with manual device management:

```python
model.to(device)
for batch in train_dataloader:
    optimizer.zero_grad()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Accelerated Trainer

HuggingFace Accelerate handles distributed training automatically:

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Prepare for distributed training
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Training loop (no manual device management needed)
for batch in train_dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    accelerator.backward(outputs.loss)  # Distributed backward
    optimizer.step()
```

### ZeRO Optimization

DeepSpeed ZeRO stages for memory efficiency:

| Stage | What's Partitioned | Memory Savings |
|-------|-------------------|----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~Nx (N = # GPUs) |

```python
from accelerate.utils import DeepSpeedPlugin

deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,  # ZeRO Stage 2
)

accelerator = Accelerator(
    log_with="wandb",
    deepspeed_plugin=deepspeed_plugin
)
```

### Distributed Evaluation

When data is spread across processes, gather before computing metrics:

```python
def eval(self, model, eval_dataloader):
    model.eval()
    all_preds, all_labels = [], []
    
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(-1)
        
        # Gather from all processes
        preds, refs = self.accelerator.gather_for_metrics(
            (predictions, batch['labels'])
        )
        all_preds.append(preds)
        all_labels.append(refs)
    
    return accuracy_metric.compute(
        predictions=torch.cat(all_preds),
        references=torch.cat(all_labels)
    )
```

## Weights & Biases Integration

Track experiments with W&B:

```python
import wandb

wandb.login()
wandb.init(project="distributed-training")

# Log metrics during training
wandb.log({'loss': loss.item(), 'epoch': epoch})

# With Accelerate
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="my-project")
accelerator.log({'loss': loss.item()})
accelerator.end_training()
```

## Project Structure

```
├── requirements.txt
└── src/
    ├── data/
    │   ├── data_connecting.py   # Dataset loading
    │   └── data_processing.py   # Tokenization pipeline
    ├── ml/
    │   ├── model.py             # Model loading
    │   └── training.py          # BasicTrainer & AcceleratedTrainer
    ├── training_application.py  # Main entry point
    └── train_sagemaker.py       # AWS SageMaker deployment
```

## Usage

### Local Training

```bash
# Basic training
python training_application.py --training_type basic

# Accelerated training
python training_application.py --training_type accelerated
```

### Multi-GPU with Accelerate

```bash
# Configure accelerate
accelerate config

# Launch distributed training
accelerate launch training_application.py --training_type accelerated
```

### AWS SageMaker

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    source_dir=".",
    entry_point="src/training_application.py",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role="arn:aws:iam::...",
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:...",
    hyperparameters={
        'training_type': "accelerated"
    }
)

huggingface_estimator.fit()
```

## Key Concepts

### Accelerate Lifecycle

1. **`accelerator.prepare()`** — Wraps model, optimizer, dataloaders for distributed training
2. **`accelerator.backward()`** — Handles gradient synchronization
3. **`accelerator.gather_for_metrics()`** — Collects data from all processes
4. **`accelerator.unwrap_model()`** — Unwraps for saving
5. **`accelerator.end_training()`** — Cleanup

### DataLoader Requirements

For HuggingFace Trainer compatibility:
- Column names: `labels`, `input_ids`, `attention_mask`
- Format: PyTorch tensors (`dataset.set_format(type='torch')`)

## Environment Variables

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
```

## References

- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Weights & Biases](https://wandb.ai/)
- [SageMaker HuggingFace Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)

