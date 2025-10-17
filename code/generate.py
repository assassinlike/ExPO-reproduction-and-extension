import json, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
def load_model(model_path, device='cuda'):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    model.eval()
    return model, tok
def generate_one(model, tok, prompt, max_new_tokens=256):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, temperature=0.8)
    return tok.decode(gen[0], skip_special_tokens=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", default="preds.jsonl")
    args = parser.parse_args()
    model, tok = load_model(args.model)
    with open(args.prompts) as f, open(args.out, "w") as w:
        for line in f:
            obj = json.loads(line)
            instruction = obj["prompt"]
            out = generate_one(model, tok, instruction)
            new_obj = {
                "instruction": instruction,
                "output": out
            }
            w.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
