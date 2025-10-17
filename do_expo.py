import os
import argparse
import gc
from safetensors import safe_open
from safetensors.torch import save_file
import torch
def get_safetensors_files(folder):
    files = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".safetensors"):
            files.append(os.path.join(folder, f))
    return files
def expo_apply(sft_dir, dpo_dir, alpha, out_dir):
    sft_files = get_safetensors_files(sft_dir)
    dpo_files = get_safetensors_files(dpo_dir)
    os.makedirs(out_dir, exist_ok=True)
    total = len(dpo_files)
    for i, (sft_file, dpo_file) in enumerate(zip(sft_files, dpo_files), 1):
        shard_sd = {}
        with safe_open(sft_file, framework="pt", device="cpu") as sft_f, \
             safe_open(dpo_file, framework="pt", device="cpu") as dpo_f:
            sft_keys = set(sft_f.keys())
            for key in dpo_f.keys():
                v1 = dpo_f.get_tensor(key).float()
                if key in sft_keys:
                    v0 = sft_f.get_tensor(key).float()
                    v2 = v1 + alpha * (v1 - v0)
                else:
                    v2 = v1
                shard_sd[key] = v2
        #保存分片
        out_file = os.path.join(out_dir, os.path.basename(dpo_file))
        save_file(shard_sd, out_file)
        del shard_sd
        gc.collect()
    print(f"Extrapolated model saved to {out_dir} (sharded)")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out_dir", default="expo_model") 
    args = parser.parse_args()

    expo_apply(args.sft, args.aligned, args.alpha, args.out_dir)
