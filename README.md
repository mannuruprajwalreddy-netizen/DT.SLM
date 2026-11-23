# NetflixBot: a Small Language Model for Netflix Movies & TV Shows

Fine-tuned **Qwen2.5-0.5B-Instruct** (LoRA) on question-answer pairs generated from the Kaggle *Netflix Movies and TV Shows* dataset. Runs locally through **Ollama**.

- Model: https://huggingface.co/YOUR_HF_USERNAME/netflix-slm
- Dataset: https://huggingface.co/datasets/YOUR_HF_USERNAME/netflix-qa-dataset

## Pipeline
Data collection → Cleaning → Instruction dataset → LoRA fine-tuning → Evaluation → Merge → GGUF → Ollama demo

## Repository layout
```
01_build_dataset.py   # CSV -> train/val JSONL + stats
02_train_lora.py      # LoRA fine-tuning + evaluation (metrics.json)
03_merge_export.py    # merge adapter, push to Hugging Face
Modelfile             # Ollama import file
MODEL_CARD.md / DATASET_CARD.md
results/              # metrics.json, loss_log.json, screenshots
```

## Step-by-step
**1. Data collection**: Download `netflix_titles.csv` from Kaggle ("Netflix Movies and TV Shows", shivamb). Place it in the repo root.

**2. Build dataset**
```bash
pip install pandas
python 01_build_dataset.py netflix_titles.csv
```
Cleans the data (drops rows without title/description, de-duplicates titles), makes ~9 Q&A types per title plus genre recommendations and out-of-scope refusals, and splits 90/10 **by title** (no leakage).

**3. Train (Google Colab, T4 GPU)**
```bash
pip install -q transformers peft datasets accelerate
python 02_train_lora.py
```
LoRA r=16, alpha=32, lr 2e-4, 2 epochs, max length 256, fp16. Reports validation loss, perplexity and per-question-type accuracy with a 95% confidence interval.

**4. Merge and upload to Hugging Face**
```bash
huggingface-cli login
python 03_merge_export.py YOUR_HF_USERNAME
```

**5. Convert to GGUF**
```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py out/netflix-slm-merged --outfile netflix-slm-q8_0.gguf --outtype q8_0
```

**6. Ollama chat demo** (install from ollama.com)
```bash
ollama create netflixbot -f Modelfile
ollama run netflixbot
>>> Who directed Black Mirror: Bandersnatch?
>>> Recommend some Documentaries.
```

## Results
See `results/metrics.json` (fill your numbers here after training).

## Limitations
Knowledge is limited to the dataset (titles up to ~2021); small model may hallucinate on unseen titles.

## License
Code: MIT. Dataset: original Kaggle license applies.
