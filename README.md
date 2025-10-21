# End-to-End QLoRA Fine-Tuning and Quantization Pipeline

This repository provides a complete workflow for fine-tuning, evaluating, merging, and quantizing large language models using **QLoRA**, followed by **adapter merging** and **post-training quantization (GPTQ or AWQ)**.  
The setup is optimized for **low-memory GPUs (like NVIDIA T4/RTX)** while maintaining efficiency, reproducibility, and model quality.

---

## Overview

This project builds a **judge model** using the **QLoRA (Quantized Low-Rank Adaptation)** approach, evaluates it, merges the trained adapters into the base model, and finally performs **post-training quantization (PTQ)** to reduce model size and memory requirements.

Each stage of the pipeline is modular and can be run independently.

---

## Repository Structure

| File | Description |
|------|--------------|
| **`01_make_payloads.py`** | Generates synthetic JSONL payloads with perturbations and edge cases for data diversity. |
| **`02_gen_summaries.py`** | Processes payloads and generates text summaries for fine-tuning or evaluation. |
| **`03_from_api_labels.py`** | Sends payloads to API/model endpoints asynchronously and retrieves labeled outputs. |
| **`04_QLoRA_Train_Judge.ipynb`** | Fine-tunes a base model using QLoRA (LoRA on quantized weights). Optimized for GPUs with limited VRAM. |
| **`05_evaluate_model.ipynb`** | Evaluates the fine-tuned model, computes metrics, and compares results with baseline or gold data. |
| **`06_merge_adapters.py`** | Merges LoRA adapter weights into the base FP16 model to produce a unified model checkpoint. |
| **`07_merge_and_quant.py`** | Performs post-training quantization using GPTQ or AWQ, optionally with calibration from your dataset. Produces a compact, deployable model. |

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

Typical dependencies:
```bash
transformers
datasets
peft
bitsandbytes
accelerate
auto-gptq
awq
torch
tqdm
aiohttp
```

---

## Workflow

### **Step 1 — Generate Payloads**
```bash
python 01_make_payloads.py --seed data/seed.json --out data/payloads.jsonl --n 500 --edge-cases
```

### **Step 2 — Generate Summaries**
```bash
python 02_gen_summaries.py --in data/payloads.jsonl --out data/summaries.jsonl
```

### **Step 3 — Fetch Labeled Responses**
```bash
python 03_from_api_labels.py --in data/summaries.jsonl --out data/labeled.jsonl --url http://localhost:4000/endpoint
```

### **Step 4 — Fine-Tune Using QLoRA**
Run **`04_QLoRA_Train_Judge.ipynb`** on Colab or a GPU-enabled environment.  
It produces adapter checkpoints under:
```
./outputs/qlora-judge-ckpt/
```

### **Step 5 — Evaluate Model**
Open **`05_evaluate_model.ipynb`** to compare the fine-tuned model’s predictions against reference labels and compute metric reports.

### **Step 6 — Merge Adapters**
```bash
python 06_merge_adapters.py --adapter ./outputs/qlora-judge-ckpt --base meta-llama/Llama-3.2-1B-Instruct --out ./merged_fp16_model
```

### **Step 7 — Quantize Model**
Perform quantization with GPTQ or AWQ:

```bash
python 07_merge_and_quant.py \
  --model_dir ./merged_fp16_model \
  --out_dir ./quantized_awq \
  --method awq \
  --bits 4 \
  --calib_jsonl ./data/validation.jsonl \
  --calib_max_samples 256 \
  --calib_max_len 2048
```

---

## Quantization Methods

| Method | Description | Calibration | Recommended For |
|---------|--------------|--------------|----------------|
| **GPTQ** | Gradient-based PTQ that minimizes weight error using representative activations. | Optional | Smaller models (1B–7B) |
| **AWQ** | Activation-aware quantization that calibrates with real inputs to preserve outlier activations. | Required | Larger or high-accuracy models |

After quantization:

```python
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("./quantized_gptq", device_map="auto")

# OR
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("./quantized_awq", device_map="auto")
```

---

## Outputs

| Stage | Output Directory | Description |
|--------|------------------|--------------|
| Fine-Tuning | `qlora-judge-ckpt/` | Adapter weights and tokenizer |
| Merge | `merged_fp16_model/` | Unified FP16 model |
| Quantization | `quantized_awq/` or `quantized_gptq/` | Compressed, deployable model |

---

## Key Features

- Efficient **parameter-efficient fine-tuning (QLoRA)**
- Dynamic sequence chunking for long inputs (6K–8K tokens)
- Safe and resumable async API calls
- Adapter merging into base FP16 model
- Calibration-aware GPTQ/AWQ quantization
- Compact, deployable quantized model artifacts

---

## Tips

- Use **AWQ** when accuracy is the priority and you have calibration data.  
- Use **GPTQ** for faster compression or limited compute.  
- Store calibration data in `.jsonl` format containing `instruction`, `input`, and `output`.  
- Always test both performance and accuracy after quantization.

---

## License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.
