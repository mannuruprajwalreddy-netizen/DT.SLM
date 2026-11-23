"""Step 2: LoRA fine-tune a small LM (Qwen2.5-0.5B-Instruct) on the Netflix Q&A data + evaluate.
Run on Colab (Runtime > T4 GPU):  pip install -q transformers peft datasets accelerate
Outputs: out/lora (adapter), out/metrics.json, out/loss_log.json
"""
import json, math, os, random, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAXLEN = 256
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

def prompt_msgs(ex):
    return [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["question"]}]

def encode(ex):
    p = tok.apply_chat_template(prompt_msgs(ex), tokenize=False, add_generation_prompt=True)
    full = p + ex["answer"] + "<|im_end|>"
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels[:len(ids)]}

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})
enc = ds.map(encode, remove_columns=ds["train"].column_names)

args = TrainingArguments(output_dir="out/ckpt", num_train_epochs=2, per_device_train_batch_size=16,
        gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=25, eval_strategy="epoch", save_strategy="no", fp16=True, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["val"],
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
trainer.train()
os.makedirs("out", exist_ok=True)
json.dump(trainer.state.log_history, open("out/loss_log.json", "w"), indent=1)
model.save_pretrained("out/lora"); tok.save_pretrained("out/lora")

# ---------- Evaluation ----------
val_loss = trainer.evaluate()["eval_loss"]
metrics = {"val_loss": val_loss, "val_perplexity": math.exp(val_loss)}

# Factual exact-match accuracy on unseen (validation) titles: key fact must appear in generated answer
model.eval()
rows = [json.loads(l) for l in open("data/val.jsonl")]
random.seed(0)
per_kind = {}
for ex in random.sample(rows, min(300, len(rows))):
    if ex["kind"] in ("OOS",): continue
    p = tok.apply_chat_template(prompt_msgs(ex), tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, do_sample=False)
    pred = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    hit = int(pred.lower() == ex["answer"].lower() or ex["answer"].lower()[:60] in pred.lower())
    k = per_kind.setdefault(ex["kind"], []); k.append(hit)
metrics["accuracy_by_question_type"] = {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in per_kind.items()}
allhits = [h for v in per_kind.values() for h in v]
acc = sum(allhits) / len(allhits); n = len(allhits)
se = math.sqrt(acc * (1 - acc) / n)
metrics["overall_accuracy"] = acc; metrics["n_eval"] = n
metrics["accuracy_95ci"] = [acc - 1.96 * se, acc + 1.96 * se]
json.dump(metrics, open("out/metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
