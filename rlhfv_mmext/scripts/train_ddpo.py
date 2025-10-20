"""
用CLIP提取图像特征, 线性映射后作为LLM的前缀embedding
用预处理数据集的前20%训练3个epoch
"""
import os
import math
import torch
import random
from torch import nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPProcessor, CLIPModel,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType
#配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.expanduser("~/rlhfv")
CLIP_PATH = os.path.join(ROOT, "models/clip-vit-base-patch16")
LLM_PATH = os.path.join(ROOT, "models/open_llama_3b_v2")
DATA_PATH = os.path.join(ROOT, "data/processed")
OUTPUT_DIR = os.path.join(ROOT, "output/ddpo_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

learning_rate = 2e-6
num_train_epochs = 3
global_batch_size = 8
micro_batch = 2
gradient_accumulation_steps = max(1, global_batch_size // micro_batch)
weight_decay = 0.01
warmup_ratio = 0.03
beta = 0.1
gamma = 2.0

lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

PREFIX_TOKENS = 1
max_length_prompt = 256
max_length_response = 256

torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dpo_loss(logpi_w, logpi_l, logpref_w, logpref_l, beta=0.1):
    z = beta * ((logpi_w - logpref_w) - (logpi_l - logpref_l))
    loss = torch.nn.functional.softplus(-z)
    return loss.mean()
#图像特征——>LLM隐藏层
class ImageToHidden(nn.Module):
    def __init__(self, clip_dim, hidden_size, prefix_tokens=1):
        super().__init__()
        self.proj = nn.Linear(clip_dim, hidden_size * prefix_tokens)
        self.prefix_tokens = prefix_tokens

    def forward(self, clip_feats):
        out = torch.tanh(self.proj(clip_feats))
        B = out.shape[0]
        hidden_size = out.shape[1] // self.prefix_tokens
        return out.view(B, self.prefix_tokens, hidden_size)
#计算batch的logπ(y|x)
def compute_logpi_for_batch(model, tokenizer, prefix_embeds, prompt_ids, prompt_mask,
                            response_ids, response_mask, seg_mask=None, gamma=1.0):
    B = prompt_ids.size(0)
    device = prefix_embeds.device
    dtype = prefix_embeds.dtype

    embed_layer = model.get_input_embeddings()
    concat_ids = torch.cat([prompt_ids, response_ids], dim=1)
    concat_mask = torch.cat([prompt_mask, response_mask], dim=1)
    token_embeds = embed_layer(concat_ids.to(model.device)).to(dtype)

    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
    prefix_len = prefix_embeds.size(1)
    attn_prefix = torch.ones((B, prefix_len), device=device, dtype=concat_mask.dtype)
    attn_mask = torch.cat([attn_prefix, concat_mask.to(device)], dim=1).long()

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, return_dict=True)
    log_probs_all = torch.log_softmax(outputs.logits, dim=-1)

    P_plus_R = concat_ids.size(1)
    max_resp = response_ids.size(1)
    prompt_lens = prompt_mask.sum(dim=1).long()
    resp_lens = response_mask.sum(dim=1).long()

    idx_j = torch.arange(0, max_resp, device=device).unsqueeze(0).expand(B, -1)
    base = (prefix_len + prompt_lens - 1).unsqueeze(1)
    pos_indices = (base + (idx_j + 1)).clamp(0, log_probs_all.size(1) - 1)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_resp)
    selected_log_probs = log_probs_all[batch_idx, pos_indices]
    labels = response_ids.to(device).long()
    token_log_probs = selected_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    resp_mask_bool = response_mask.to(device).float()
    token_log_probs = token_log_probs * resp_mask_bool

    if seg_mask is None:
        weights = resp_mask_bool
    else:
        seg = seg_mask.to(device).float() * resp_mask_bool
        weights = resp_mask_bool * (1.0 - seg) + seg * gamma

    weighted_sum = (token_log_probs * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return weighted_sum / denom
#数据整理
def collate_samples(samples, tokenizer, clip_processor, max_p=max_length_prompt, max_r=max_length_response):
    B = len(samples)
    prompts, chosens, rejecteds = [], [], []
    chosen_seg_masks, rejected_seg_masks, image_paths = [], [], []

    for s in samples:
        prompt = s.get("prompt") if isinstance(s.get("prompt"), str) else (s.get("instruction") or "")
        prompts.append(prompt)
        chosens.append(s.get("chosen_response") or "")
        rejecteds.append(s.get("rejected_response") or "")
        chosen_seg_masks.append(s.get("chosen_mask"))
        rejected_seg_masks.append(s.get("rejected_mask"))
        image_paths.append(s.get("image_path") or s.get("image") or "")

    tokenized_prompts = tokenizer(prompts, padding=True, truncation=True, max_length=max_p, return_tensors="pt")
    tokenized_chosen = tokenizer(chosens, padding=True, truncation=True, max_length=max_r, return_tensors="pt")
    tokenized_rejected = tokenizer(rejecteds, padding=True, truncation=True, max_length=max_r, return_tensors="pt")

    def normalize_seg_mask_list(seg_list, tokenized):
        if all(s is None for s in seg_list):
            return None
        res = []
        for i, s in enumerate(seg_list):
            L = tokenized["input_ids"].size(1)
            if s is None:
                res.append([0] * L)
            else:
                arr = list(s)
                if len(arr) >= L:
                    res.append(arr[:L])
                else:
                    res.append(arr + [0] * (L - len(arr)))
        return torch.tensor(res, dtype=torch.long)

    chosen_seg_mask_tensor = normalize_seg_mask_list(chosen_seg_masks, tokenized_chosen)
    rejected_seg_mask_tensor = normalize_seg_mask_list(rejected_seg_masks, tokenized_rejected)

    return {
        "prompt_ids": tokenized_prompts["input_ids"],
        "prompt_mask": tokenized_prompts["attention_mask"],
        "chosen_ids": tokenized_chosen["input_ids"],
        "chosen_mask": tokenized_chosen["attention_mask"],
        "rejected_ids": tokenized_rejected["input_ids"],
        "rejected_mask": tokenized_rejected["attention_mask"],
        "chosen_seg_mask": chosen_seg_mask_tensor,
        "rejected_seg_mask": rejected_seg_mask_tensor,
        "image_paths": image_paths
    }
#主训练逻辑
def main():
    print("加载数据集")
    if os.path.isdir(DATA_PATH) and ("train" in os.listdir(DATA_PATH) or "dataset_dict.json" in os.listdir(DATA_PATH)):
        ds_full = load_from_disk(DATA_PATH)
        ds = ds_full["train"] if "train" in ds_full else ds_full
    else:
        raise FileNotFoundError(f"找不到数据集路径{DATA_PATH}")
    print(f"加载{len(ds)}条样本")

    subset_size = max(1, int(len(ds) * 0.2))
    ds_small = ds.select(range(subset_size))
    print(f"用前20%({len(ds_small)}条)数据训练")

    print("加载CLIP模型")
    clip_model = CLIPModel.from_pretrained(CLIP_PATH).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    clip_dim = getattr(clip_model.visual_projection, "in_features", None) or getattr(clip_model.config, "projection_dim", 768)
    print("CLIP嵌入维度为", clip_dim)

    print("加载LLM")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("自动将 pad_token 设为 eos_token")

    llm_dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    policy_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype=llm_dtype, low_cpu_mem_usage=True).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype=llm_dtype, low_cpu_mem_usage=True).to(DEVICE)
    ref_model.eval()
    llm_hidden = policy_model.config.hidden_size
    print("LLM隐藏层维度为", llm_hidden)

    image_to_hidden = ImageToHidden(clip_dim, llm_hidden, prefix_tokens=PREFIX_TOKENS).to(DEVICE).to(policy_model.dtype)

    print("注入LoRA模块")
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    policy_model = get_peft_model(policy_model, lora_config).to(DEVICE)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_updates = math.ceil(len(ds_small) / micro_batch) * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * warmup_ratio), total_updates)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    def dataset_iterator():
        cur = []
        for ex in ds_small:
            cur.append(ex)
            if len(cur) >= micro_batch:
                yield cur
                cur = []
        if cur:
            yield cur

    print("\n开始训练")
    step = 0
    policy_model.train()

    for epoch in range(num_train_epochs):
        print(f"\nEpoch {epoch+1}/{num_train_epochs}")
        running_loss = 0.0
        for batch_samples in tqdm(dataset_iterator(), total=math.ceil(len(ds_small)/micro_batch)):
            batch = collate_samples(batch_samples, tokenizer, clip_processor, max_length_prompt, max_length_response)
            img_inputs = clip_processor(images=[Image.open(p).convert("RGB") for p in batch["image_paths"]], return_tensors="pt")
            for k in img_inputs: img_inputs[k] = img_inputs[k].to(DEVICE)

            with torch.no_grad():
                clip_feats = clip_model.get_image_features(**img_inputs).detach().to(policy_model.dtype).to(DEVICE)
            prefix_embeds = image_to_hidden(clip_feats).to(policy_model.dtype)

            prompt_ids = batch["prompt_ids"].to(DEVICE)
            prompt_mask = batch["prompt_mask"].to(DEVICE)
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)
            chosen_seg = batch["chosen_seg_mask"].to(DEVICE) if batch["chosen_seg_mask"] is not None else None
            rejected_seg = batch["rejected_seg_mask"].to(DEVICE) if batch["rejected_seg_mask"] is not None else None

            try:
                with torch.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    logpi_w = compute_logpi_for_batch(policy_model, tokenizer, prefix_embeds, prompt_ids, prompt_mask,
                                                      chosen_ids, chosen_mask, chosen_seg, gamma=gamma)
                    logpi_l = compute_logpi_for_batch(policy_model, tokenizer, prefix_embeds, prompt_ids, prompt_mask,
                                                      rejected_ids, rejected_mask, rejected_seg, gamma=gamma)
                    with torch.no_grad():
                        logpref_w = compute_logpi_for_batch(ref_model, tokenizer, prefix_embeds, prompt_ids, prompt_mask,
                                                            chosen_ids, chosen_mask, chosen_seg, gamma=gamma)
                        logpref_l = compute_logpi_for_batch(ref_model, tokenizer, prefix_embeds, prompt_ids, prompt_mask,
                                                            rejected_ids, rejected_mask, rejected_seg, gamma=gamma)
                    loss = dpo_loss(logpi_w, logpi_l, logpref_w, logpref_l, beta=beta)

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()
                step += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA OOM! 清理显存并跳过该batch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler = torch.amp.GradScaler("cuda")
                    continue
                else:
                    raise

        avg_loss = running_loss / max(1, math.ceil(len(ds_small)/micro_batch))
        print(f"Epoch{epoch+1}平均loss:{avg_loss:.6f}")

        out_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        os.makedirs(out_path, exist_ok=True)
        policy_model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)
        print("模型保存到", out_path)

    print("finish!")

if __name__ == "__main__":
    main()
