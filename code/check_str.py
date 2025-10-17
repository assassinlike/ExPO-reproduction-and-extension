# check_str.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, sys
from transformers import AutoModelForCausalLM
def load_keys(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True
    )
    ks = set(model.state_dict().keys())
    del model
    return ks
if __name__ == "__main__":
    sft_path = "/root/expo_project/models/sft"
    aligned_path = sys.argv[1] if len(sys.argv) > 1 else "/root/expo_project/models/dpo"
    m0 = load_keys(sft_path)
    m1 = load_keys(aligned_path)
    #print(len(m0)," ",len(m1))
    print("only m0:", sorted(list(m0 - m1))[:10])
    print("only m1:", sorted(list(m1 - m0))[:10])
