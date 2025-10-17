import re
import argparse
import torch
import os
from safetensors import safe_open
from safetensors.torch import save_file
def load_sd(path):
    if os.path.isdir(path):
        full_path = os.path.join(path, "model.safetensors")
        if os.path.exists(full_path):
            path = full_path
        else:
            raise FileNotFoundError(f"{path}中没有.safetensors")
    if not os.path.isfile(path) or not path.endswith(".safetensors"):
        raise ValueError(f"{path}不是.safetensors")
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).cpu()
    return sd
def detect_num_layers(sd):
    indices = set()
    for k in sd.keys():
        m = re.search(r'model\.layers\.(\d+)\.', k) #mistral的结构
        if m:
            indices.add(int(m.group(1)))
    return max(indices) + 1 if indices else None
def group_of_key(k, num_layers):
    if num_layers is None or num_layers <= 0:
        return 'others'
    m = re.search(r'model\.layers\.(\d+)\.', k)
    if m:
        idx = int(m.group(1))
        if idx < num_layers // 3:
            return 'early'
        elif idx < 2 * num_layers // 3:
            return 'mid'
        else:
            return 'late'
    else:
        return 'others'
def expo_layer(sd0, sd1, alpha_map):
    sd2 = {}
    num_layers = detect_num_layers(sd1)
    for k, v1 in sd1.items():
        grp = group_of_key(k, num_layers)
        alpha = alpha_map.get(grp, 1.0)
        if k in sd0:
            v0 = sd0[k].float()
            v1f = v1.float()
            sd2[k] = (v1f + alpha * (v1f - v0)).to(torch.float32)
        else:
            #sft里没有就保留原本的
            sd2[k] = v1.float()
    return sd2
def save_safetensors(sd, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cpu_sd = {k: v.cpu() for k, v in sd.items()}
    save_file(cpu_sd, out_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--alpha_early", type=float, default=1.0)
    parser.add_argument("--alpha_mid", type=float, default=1.0)
    parser.add_argument("--alpha_late", type=float, default=1.0)
    parser.add_argument("--out", default="expo_layerwise.safetensors")
    args = parser.parse_args()
    if not args.out.endswith(".safetensors"):
        args.out += ".safetensors"
    print("加载sft")
    sd0 = load_sd(args.sft)
    print("加载dpo")
    sd1 = load_sd(args.aligned)
    alpha_map = {
        'early': args.alpha_early,
        'mid': args.alpha_mid,
        'late': args.alpha_late,
        'others': 1.0,
    }
    print("正常运行")
    sd2 = expo_layer(sd0, sd1, alpha_map)
    save_safetensors(sd2, args.out)
    print("finish!")