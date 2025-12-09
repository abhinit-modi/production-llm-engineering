# Production LLM Engineering

A comprehensive collection of projects demonstrating end-to-end Large Language Model (LLM) development — from building transformer architectures from scratch to deploying production-ready RAG applications.

## Projects

### [01 - Transformer Architecture](./01-transformer-architecture/)
**Building LLM Components from Scratch**

Custom PyTorch implementations of modern transformer architecture components:
- **Rotary Position Embeddings (RoPE)** — Position encoding that enables better length generalization
- **Sliding Window Attention** — Memory-efficient attention with configurable window sizes
- **Mixture of Experts (MoE)** — Sparse expert routing with top-k gating
- **SiGLU Activation** — Gated linear units for improved FFN performance
- **RMSNorm** — Efficient layer normalization

### [02 - LLM Fine-tuning & Alignment](./02-llm-finetuning-alignment/)
**Advanced Training Techniques for Language Models**

Implementation of multiple alignment strategies:
- **Supervised Fine-tuning (SFT)** — Instruction tuning on custom datasets
- **RLHF** — Reinforcement Learning from Human Feedback pipeline
- **DPO** — Direct Preference Optimization for simpler alignment
- **ORPO** — Odds Ratio Preference Optimization
- **Model Evaluation** — Automated evaluation pipelines with HuggingFace

### [03 - Distributed Training](./03-distributed-training/)
**Scalable Training Infrastructure**

Production-ready distributed training setup:
- **HuggingFace Accelerate** — Multi-GPU training with minimal code changes
- **ZeRO Optimizer** — Memory-efficient distributed optimization (ZeRO-1/2/3)
- **Weights & Biases** — Experiment tracking and visualization
- **AWS SageMaker** — Cloud training infrastructure

### [04 - Scalable Attention](./04-scalable-attention/)
**Efficient Attention Mechanisms for Long Sequences**

Architectural modifications for scalability:
- **Sliding Window Attention** — O(n × window) complexity vs O(n²)
- **Grouped Query Attention** — Reduced KV-cache memory footprint
- **RoPE Scaling** — Extended context length support
- **Dynamic Vocabulary Expansion** — Adding special tokens at runtime

### [05 - Model Deployment](./05-model-deployment/)
**Production Model Serving with vLLM**

High-performance inference infrastructure:
- **vLLM Deployment** — Optimized inference with continuous batching
- **FastAPI Streaming** — Server-Sent Events (SSE) for real-time generation
- **OpenAI-Compatible Server** — Drop-in replacement API
- **Docker Containerization** — Production-ready deployment configs

### [06 - RAG Application](./06-rag-application/)
**Full-Stack Retrieval-Augmented Generation**

End-to-end conversational AI system:
- **Backend (FastAPI)**
  - Streaming chat endpoints with SSE
  - Vector database integration for RAG
  - Hybrid search (semantic + keyword)
  - Chat history persistence with SQLAlchemy
  - Admin dashboard with SQLAdmin
- **Frontend (Streamlit)**
  - Real-time streaming chat interface
  - Multiple chat modes (simple, history, RAG)
  - User session management

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, Transformers, PEFT, TRL |
| **Training** | Accelerate, DeepSpeed, Weights & Biases |
| **Inference** | vLLM, FastAPI, Streaming |
| **Data** | HuggingFace Datasets, Vector DBs |
| **Infrastructure** | Docker, AWS SageMaker |
| **Frontend** | Streamlit |

## Getting Started

Each project contains its own `requirements.txt`. To run a specific project:

```bash
cd <project-folder>
pip install -r requirements.txt
```
