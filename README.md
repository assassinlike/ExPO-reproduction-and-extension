# ExPO Reproduction and Extension — Layerwise and Multimodal Alignment

> **Model Extrapolation Expedites Alignment — Reproduction and Extension**  
> **Author:** Yi-He Zang (臧一赫)  
> **Date:** October 2025  

---

## Overview

This repository reproduces and extends the **Model Extrapolation (ExPO)** method proposed by **CoAI** in the paper *“Model Extrapolation Expedites Alignment”*.  
Our work includes:

- Full **reproduction** of the original ExPO alignment framework.  
- Proposal of a new **Layerwise Extrapolation** method to improve generation stability.  
- Preliminary **multimodal extension** of ExPO applied to the **RLHF-V** dataset to mitigate hallucination.

Code and results are available here:  
 [https://github.com/assassinlike/ExPO-reproduction-and-extension](https://github.com/assassinlike/ExPO-reproduction-and-extension)

---

##  1. Reproduction Setup

### **Model and Data**

| Component               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **SFT Model**           | `zephyr-7b-sft-full` (same as original paper)                |
| **Aligned Model (DPO)** | `chujiezheng/zephyr_0.05` (trained with only 5% of the DPO dataset) |
| **Evaluation Set**      | AlpacaEval                                                   |
| **Hardware**            | Alibaba Cloud GPU (NVIDIA A10)                               |

### **Extrapolation Parameters**

| Item                          | Value / Note                           |
| ----------------------------- | -------------------------------------- |
| Extrapolation coefficient `α` | 12 (corresponding to 5% training data) |
| Memory control                | Chunked model loading to prevent OOM   |

---

##  2. Reproduction Results

- The reproduced ExPO model exhibits consistent **instruction-following** ability.  
- Some samples show **instruction repetition before answering** (a known ExPO behavior).  
- A few cases display **instruction-only** outputs, addressed by our Layerwise improvement.

---

##  3. Layerwise Extrapolation (Proposed)

### **Motivation**

Direct extrapolation across all transformer layers can be unstable.  
We propose applying different extrapolation coefficients per layer group.

| Layer Group   | Range      | α Value |
| ------------- | ---------- | ------- |
| Early Layers  | First 1/3  | 5       |
| Middle Layers | Middle 1/3 | 10      |
| Late Layers   | Last 1/3   | 15      |

### **Results**

-  Improved stability — no “instruction-only” outputs observed.  
-  Higher coherence and naturalness in responses.  
-  Some repetition behavior remains.

---

##  4. Multimodal Extension (RLHF-V)

We extend ExPO into **vision-language alignment** using the **RLHF-V** framework  
([Paper: *RLHF-V: Towards Trustworthy MLLMs via Fine-grained Correctional Human Feedback*](https://arxiv.org/abs/2312.00849)).

### **Setup**

| Component          | Description                                              |
| ------------------ | -------------------------------------------------------- |
| **Vision Encoder** | `clip-vit-base-patch16`                                  |
| **LLM Backbone**   | `open_llama_3b_v2`                                       |
| **Training Data**  | 20% of RLHF-V dataset                                    |
| **Fusion Method**  | Linear projection from CLIP embedding → LLM hidden space |
| **Loss Function**  | DPO + segment-weighted correction (`γ = 2.0`)            |

### **Key Parameters**

| Parameter           | Value | Note                      |
| ------------------- | ----- | ------------------------- |
| Learning rate       | 2e-6  |                           |
| Epochs              | 3     |                           |
| Global batch size   | 8     |                           |
| Micro batch         | 2     | Gradient accumulation = 4 |
| Weight decay        | 0.01  |                           |
| Warmup ratio        | 0.03  |                           |
| β                   | 0.1   | DPO temperature           |
| γ                   | 2.0   | Segment weighting         |
| max_length_prompt   | 256   |                           |
| max_length_response | 256   |                           |

---

##  5. Training Logic (DDPO Framework)

Implemented in `train_ddpo.py`:

1. Load HuggingFace-style dataset (20% subset).  
2. Extract CLIP embeddings → project to LLM hidden space.  
3. Concatenate image prefix embedding + text tokens.  
4. Compute log probabilities for chosen/rejected pairs.  
5. Apply DPO loss with segment weighting (γ).  
6. Update LoRA adapters via gradient accumulation.  
7. Save LoRA adapter weights and tokenizer per epoch.  
8. AMP + OOM recovery ensures training stability.

---

##  6. Current Status and Issues

-  Multimodal DDPO training encountered **tensor shape mismatches** and **mixed precision scaling errors**.  
-  Inference pipeline (`init_infer.py`) successfully verified **dual-modality input** (image + prompt).  
-  Preprocessing pipeline (`preprocess_rlhfv.py`) converts raw RLHF-V JSONs into `DatasetDict` format for training.

---

##  7. Evaluation and Future Work

### **Limitations**

- Could not perform GPT-4-turbo automatic evaluation (due to account constraints).  
- Lack of win-rate metric.

### **Next Steps**

- Use open-source evaluation models for objective alignment assessment.  
- Refine layer grouping in Layerwise Extrapolation.  
- Introduce regularization or prompt-control strategies to reduce repetition.  
- Fix multimodal DDPO training bugs and reapply ExPO extrapolation.

---

##  8. Preprocessing Overview

Script: `scripts/preprocess_rlhfv.py`

Output structure:

```
~/rlhfv/data/processed/
│
├── dataset_info.json
├── train/
│   ├── data.arrow
│   └── state.json
└── ...
```

Each sample includes:

```json
{
  "prompt": "...",
  "chosen_response": "...",
  "rejected_response": "...",
  "image_path": "...",
  "chosen_mask": null,
  "rejected_mask": null
}
```

---

##  9. Conclusion

1.  Successfully reproduced **ExPO** from CoAI.  
2.  Verified that even **5% DPO training** can be extrapolated with a large α.  
3.  Proposed **Layerwise Extrapolation** for enhanced stability.  
4.  Built a **multimodal DDPO training framework**, verified real **image+text** input at inference time.

---

##  Reference

- CoAI, *Model Extrapolation Expedites Alignment*, 2024.  
- RLHF-V, *Towards Trustworthy MLLMs via Fine-grained Correctional Human Feedback*, 2023.  

---

## Repository Structure

```
ExPO-reproduction-and-extension/
│
├── scripts/
│   ├── do_expo.py
│   ├── expo_layerwise.py
│   ├── init_infer.py
│   ├── preprocess_rlhfv.py
│   ├── train_ddpo.py
│
├── data/
│   ├── RLHF-V-Dataset/
│   └── processed/
│
├── results/
│   ├── expo_pred_dpo_train0.05.jsonl
│   └── expo_50_preds.jsonl
│
└── README.md
```

---

> **Contact:**  
> For technical questions or collaboration inquiries, please reach out via GitHub Issues.

---
