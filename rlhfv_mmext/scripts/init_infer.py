#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
init_infer.py
多模态推理脚本（图片 + prompt）：
- 使用 CLIP 提取图像特征 -> 线性投影 -> 作为 prefix_embeds 输入 LLM
- 支持命令行指定图片和 prompt
- 支持 LoRA adapter
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
from PIL import Image
import argparse
from peft import PeftModel

# ----------------------------
# 配置
# ----------------------------
ROOT = os.path.expanduser("~/rlhfv")
LLM_PATH = os.path.join(ROOT, "models/open_llama_3b_v2")
CLIP_PATH = os.path.join(ROOT, "models/clip-vit-base-patch16")
PREFIX_TOKENS = 1  # 用 1 个 token 表示 image embedding
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 命令行参数
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="图片文件路径")
parser.add_argument("--prompt", type=str, required=True, help="文本指令 prompt")
parser.add_argument("--lora", type=str, default=None, help="LoRA adapter 目录，如果有")
args = parser.parse_args()

image_path = os.path.expanduser(args.image)
prompt = args.prompt

if not os.path.exists(image_path):
    raise FileNotFoundError(f"图片 {image_path} 不存在")

print("🖼️ 图像路径:", image_path)
print("💬 Prompt:", prompt)
print("🚀 加载 CLIP 与 LLM 模型...")

# ----------------------------
# 加载 CLIP
# ----------------------------
clip_model = CLIPModel.from_pretrained(CLIP_PATH).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)

# ----------------------------
# 加载 LLM
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32)
llm.to(DEVICE)

# 如果有 LoRA adapter
if args.lora and os.path.exists(args.lora):
    print("⚡ 加载 LoRA adapter...")
    llm = PeftModel.from_pretrained(llm, args.lora)
else:
    print("⚠️ 未找到 LoRA adapter，使用基础模型")

llm.eval()

# ----------------------------
# Image -> prefix embedding
# ----------------------------
class ImageToHidden(torch.nn.Module):
    def __init__(self, clip_dim, hidden_size, prefix_tokens=1):
        super().__init__()
        self.proj = torch.nn.Linear(clip_dim, hidden_size * prefix_tokens)
        self.prefix_tokens = prefix_tokens

    def forward(self, clip_feats):
        out = self.proj(clip_feats)
        out = torch.tanh(out)
        B = out.shape[0]
        hidden_size = out.shape[1] // self.prefix_tokens
        out = out.view(B, self.prefix_tokens, hidden_size)
        return out

# ----------------------------
# 推理函数
# ----------------------------
def infer(image_path, prompt):
    # 读取图片
    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        clip_feats = clip_model.get_image_features(**img_inputs)
        clip_feats = clip_feats.to(llm.dtype)

    # 动态获取 CLIP 输出维度
    clip_dim = clip_feats.shape[1]
    image_to_hidden = ImageToHidden(clip_dim, llm.config.hidden_size, prefix_tokens=PREFIX_TOKENS).to(DEVICE).to(llm.dtype)
    prefix_embeds = image_to_hidden(clip_feats)  # [1, prefix, hidden]

    # 构造输入 token
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    # 拼接 prefix_embeds
    with torch.no_grad():
        outputs = llm.generate(
            input_ids=None,
            inputs_embeds=torch.cat([prefix_embeds, llm.get_input_embeddings()(input_ids)], dim=1),
            attention_mask=torch.cat([torch.ones(prefix_embeds.size()[:-1], device=DEVICE), attention_mask], dim=1),
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ----------------------------
# 执行
# ----------------------------
if __name__ == "__main__":
    result = infer(image_path, prompt)
    print("\n=== Generated Answer ===")
    print(result)
