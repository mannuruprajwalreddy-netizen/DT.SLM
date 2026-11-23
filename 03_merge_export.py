"""Step 3: Merge LoRA into base, save full model, (optionally) push model + dataset to Hugging Face.
Usage: python 03_merge_export.py YOUR_HF_USERNAME   (run `huggingface-cli login` first; skip push by passing nothing)
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
merged = PeftModel.from_pretrained(base, "out/lora").merge_and_unload()
tok = AutoTokenizer.from_pretrained("out/lora")
merged.save_pretrained("out/netflix-slm-merged"); tok.save_pretrained("out/netflix-slm-merged")
print("Merged model saved to out/netflix-slm-merged")

if len(sys.argv) > 1:
    from huggingface_hub import HfApi
    user = sys.argv[1]; api = HfApi()
    api.create_repo(f"{user}/netflix-slm", exist_ok=True)
    api.upload_folder(folder_path="out/netflix-slm-merged", repo_id=f"{user}/netflix-slm")
    api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md", repo_id=f"{user}/netflix-slm")
    api.create_repo(f"{user}/netflix-qa-dataset", repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path="data", repo_id=f"{user}/netflix-qa-dataset", repo_type="dataset")
    api.upload_file(path_or_fileobj="DATASET_CARD.md", path_in_repo="README.md",
                    repo_id=f"{user}/netflix-qa-dataset", repo_type="dataset")
    print("Pushed to Hugging Face.")
