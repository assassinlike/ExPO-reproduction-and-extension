# preprocess_rlhfv.py
import os
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# =============================
# 路径配置
# =============================
RAW_DATA_DIR = os.path.expanduser("~/rlhfv/data/RLHF-V-Dataset")
OUTPUT_DIR = os.path.expanduser("~/rlhfv/data/processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# 遍历 JSON 文件
# =============================
data_records = []

json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".json")]

for fname in tqdm(json_files, desc="Processing JSON files"):
    fpath = os.path.join(RAW_DATA_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        item = json.load(f)

    # 提取文本和图片
    prompt = item["text"].get("question", "")
    chosen = item["text"].get("chosen", "")
    rejected = item["text"].get("rejected", "")
    image_path = os.path.join(RAW_DATA_DIR, item.get("image_path", ""))

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"⚠️  图片不存在: {image_path}, 跳过该条数据")
        continue

    record = {
        "prompt": prompt,
        "chosen_response": chosen,
        "rejected_response": rejected,
        "image_path": image_path,
        "chosen_mask": None,   # 可选分段 mask，DDPO 可直接使用 gamma 权重
        "rejected_mask": None
    }
    data_records.append(record)

# =============================
# 构造 Dataset
# =============================
dataset = Dataset.from_list(data_records)

# 可分训练/验证集（这里简单全量训练）
dataset_dict = DatasetDict({
    "train": dataset
})

# =============================
# 保存处理后的数据
# =============================
dataset_dict.save_to_disk(OUTPUT_DIR)
print(f"✅ 数据处理完成，保存到 {OUTPUT_DIR}")
