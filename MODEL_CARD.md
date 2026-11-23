---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
- YOUR_HF_USERNAME/netflix-qa-dataset
language: en
tags: [small-language-model, lora, netflix, ollama, design-thinking]
---

# netflix-slm

A ~0.5B-parameter small language model fine-tuned with LoRA to answer questions about Netflix movies and TV shows (description, director, cast, year, rating, duration, country, genre, recommendations).

## Intended use
Educational demo for the AIML-C Design Thinking course; local chat via Ollama. Not for production.

## Training
- Base: Qwen2.5-0.5B-Instruct; LoRA r=16, alpha=32, dropout 0.05, all attention+MLP projections
- 2 epochs, lr 2e-4 cosine, effective batch 32, max length 256, fp16, 1x T4 GPU
- Loss computed on answer tokens only

## Evaluation
| Metric | Value |
|---|---|
| Validation loss | FILL |
| Validation perplexity | FILL |
| Factual accuracy (unseen titles, 95% CI) | FILL |

## Limitations
Only knows the Kaggle Netflix catalogue snapshot; may hallucinate for titles outside it.

## Usage
```python
from transformers import pipeline
pipe = pipeline("text-generation", "YOUR_HF_USERNAME/netflix-slm")
```
