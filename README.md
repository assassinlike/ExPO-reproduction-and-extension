# Model Extrapolation Expedites Alignment — Reproduction and Extension

**Author:** Yihe Zang  
**Date:** October 2025  
**Paper:** [Model Extrapolation Expedites Alignment (CoAI Team)](https://arxiv.org/abs/2406.00000)  
**Repository:** [github.com/assassinlike/ExPO-reproduction-and-extension](https://github.com/assassinlike/ExPO-reproduction-and-extension)

---

## Overview

This project reproduces the **ExPO (Model Extrapolation)** method proposed by the CoAI team and extends it with a new technique — **Layerwise Extrapolation**.

We evaluate ExPO under **limited DPO data (5%)**, compared to the original settings of **0.1, 0.2, 0.4, and 1.0** training fractions.

---

## Goals

- Reproduce the original **Model Extrapolation (ExPO)** method.  
- Verify its performance under **5% DPO training data**.  
- Propose **Layerwise Extrapolation** for improved stability.

---

## Experimental Setup

| Component               | Description                          |
| ----------------------- | ------------------------------------ |
| **SFT Model**           | `zephyr-7b-sft-full` (same as paper) |
| **Aligned Model (DPO)** | `chujiezheng/zephyr_0.05`            |
| **Evaluation Set**      | AlpacaEval                           |
| **Hardware**            | Alibaba Cloud GPU (NVIDIA A10)       |

### Parameters

| Parameter          | Description                                           |
| ------------------ | ----------------------------------------------------- |
| **α Selection**    | Original: 10%→8.5, 20%→2.0; this work: α=12 (5% data) |
| **Memory Control** | Chunked loading to prevent OOM                        |

---

## Results (Reproduction)

On the **AlpacaEval** dataset, the reproduced ExPO model shows:

- Tendency to restate the instruction before answering.  
- Normal-quality responses after restatement.  
- Some cases where only restatement occurs (no actual answer).

---

## Extension: Layerwise Extrapolation

To address instability in direct extrapolation, we propose **Layerwise Extrapolation**, which applies different α values to different Transformer layer groups.

| Layer Group   | Layer Range | α Value |
| ------------- | ----------- | ------- |
| Early Layers  | First 1/3   | 5       |
| Middle Layers | Middle 1/3  | 10      |
| Late Layers   | Last 1/3    | 15      |

### Results

- Improved stability: No "restatement-only" samples.  
- Better response quality: More natural and logical answers.  
- Minor instruction restatement remains.

---

## Evaluation and Limitations

- **Limitation:** Unable to use GPT-4-turbo for automatic evaluation due to account restrictions (win rate unavailable).  
- **Future Work:**
  - Try free or web-based evaluation models.  
  - Explore finer-grained layer-wise α distribution.  
  - Use regularization or prompt tuning to reduce restatement.

---

## Conclusion

1. Successfully reproduced the CoAI ExPO method.  
2. Verified that large α can yield reasonable alignment with only 5% DPO data.  
3. Proposed **Layerwise Extrapolation**, which improves stability and quality.

---

## Repository Structure

```
ExPO-reproduction-and-extension/
│
├── code/
│   ├── do_expo.py              # (Memory-efficient)
│   ├── expo_layerwise.py       # Layerwise extrapolation
│   ├── check_str.py            
│   └── generate.py             
│
├── result/
│   ├── expo_pred_dpo_train0.05.jsonl
│   └── expo_50_preds.jsonl
│
└── README.md
```

---

## Acknowledgements

Thanks to the **CoAI Team** for open-sourcing their ExPO framework and to **Hugging Face** for model hosting.