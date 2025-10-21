#!/usr/bin/env python3
import argparse, os, json, torch, itertools
from typing import List, Dict, Any, Iterable

# -------------------------------
# Helpers
# -------------------------------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines
                continue

def extract_texts(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """
    Prefer the 'input' (your SFT uses instruction+input for the judge).
    Fallbacks try common fields if shape differs.
    """
    out = []
    for r in rows:
        if isinstance(r.get("input"), str) and r["input"].strip():
            # typical SFT row: instruction + "\n\n" + input already in "input"
            # If your rows separate instruction, merge them to be safe.
            instr = r.get("instruction", "")
            inp   = r.get("input", "")
            txt = f"{instr.strip()}\n\n{inp.strip()}".strip() if instr else inp
            out.append(txt)
        elif isinstance(r.get("text"), str) and r["text"].strip():
            out.append(r["text"].strip())
        elif isinstance(r.get("prompt"), str) and r["prompt"].strip():
            out.append(r["prompt"].strip())
        elif isinstance(r.get("instruction"), str) and r["instruction"].strip():
            out.append(r["instruction"].strip())
    # dedupe, keep order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def chunk_to_token_windows(tokenizer, text: str, max_len: int, stride: int) -> List[List[int]]:
    """
    Tokenize long text and split into overlapping token windows (max_len, stride).
    Returns list of input_ids lists.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return []
    if len(ids) <= max_len:
        return [ids]
    windows = []
    start = 0
    while start < len(ids):
        end = start + max_len
        chunk = ids[start:end]
        if not chunk:
            break
        windows.append(chunk)
        if end >= len(ids):
            break
        # slide by max_len - stride
        step = max_len - stride
        if step <= 0:
            break
        start += step
    return windows

def build_calibration_batches(tokenizer, texts: List[str], max_len: int, stride: int, max_samples: int) -> List[List[int]]:
    """
    Produce up to max_samples token sequences (<= max_len each) from texts.
    """
    seqs = []
    for t in texts:
        for win in chunk_to_token_windows(tokenizer, t, max_len=max_len, stride=stride):
            seqs.append(win)
            if len(seqs) >= max_samples:
                return seqs
    return seqs

def make_gptq_examples(token_seqs: List[List[int]], device: str):
    """
    auto-gptq typically accepts examples as list of dicts with input_ids tensors
    """
    ex = []
    for s in token_seqs:
        tens = torch.tensor([s], dtype=torch.long)  # batch=1
        if device == "cuda":
            tens = tens.cuda()
        ex.append({"input_ids": tens})
    return ex

# -------------------------------
# CLI
# -------------------------------
parser = argparse.ArgumentParser(description="Quantize model with GPTQ or AWQ using calibration JSONL.")
parser.add_argument("--model_dir", required=True, help="Merged FP16 model directory (after adapter merge).")
parser.add_argument("--out_dir",   default="./quantized", help="Output directory.")
parser.add_argument("--method",    choices=["gptq", "awq"], default="gptq", help="Quantization method.")
parser.add_argument("--bits",      type=int, default=4, help="Quantization bits (4 or 8).")
parser.add_argument("--group_size", type=int, default=128, help="Group size for GPTQ.")
parser.add_argument("--desc_act",  action="store_true", help="Use desc_act for GPTQ.")
parser.add_argument("--calib_jsonl", required=True,
                    help="Path to calibration JSONL (your SFT/eval jsonl). Uses 'instruction' + 'input'.")
parser.add_argument("--calib_max_samples", type=int, default=256, help="Max calibration token windows.")
parser.add_argument("--calib_max_len",     type=int, default=2048, help="Max tokens per calibration window.")
parser.add_argument("--calib_stride",      type=int, default=256, help="Token overlap between windows.")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 {args.method.upper()} quantization ({args.bits}-bit) on {device}")

# -------------------------------
# Tokenizer + Calibration data
# -------------------------------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"📥 Reading calibration from {args.calib_jsonl}")
texts = extract_texts(read_jsonl(args.calib_jsonl))
if not texts:
    raise SystemExit("No usable text found in the calibration JSONL. Check fields: instruction/input/text/prompt.")

print(f"🧪 Preparing calibration windows (max_len={args.calib_max_len}, stride={args.calib_stride}) ...")
calib_token_windows = build_calibration_batches(
    tokenizer, texts,
    max_len=args.calib_max_len,
    stride=args.calib_stride,
    max_samples=args.calib_max_samples
)
print(f"✅ Built {len(calib_token_windows)} calibration windows.")

# -------------------------------
# GPTQ
# -------------------------------
if args.method == "gptq":
    if device != "cuda":
        raise EnvironmentError("GPTQ quantization requires CUDA. Run on a GPU machine/Colab with T4/A100/V100.")

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    quant_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )

    print("🔩 Loading FP16 model for GPTQ ...")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_dir,
        quantize_config=quant_config,
        device_map="auto",
        use_safetensors=True
    )

    print("🧠 Running GPTQ calibration+quantization ...")
    gptq_examples = make_gptq_examples(calib_token_windows, device="cuda")
    # newer auto-gptq accepts `examples=...`; if your version doesn’t, it will still quantize with defaults.
    try:
        model.quantize(examples=gptq_examples)
    except TypeError:
        print("⚠️ Installed auto-gptq doesn’t accept `examples=`; proceeding without explicit examples.")
        model.quantize()

    model.save_quantized(args.out_dir)
    print(f"✅ GPTQ quantized model saved to: {args.out_dir}")

# -------------------------------
# AWQ
# -------------------------------
elif args.method == "awq":
    # AWQ strongly benefits from real calibration text; pass raw texts.
    from awq import AutoAWQForCausalLM

    print("🔩 Loading FP16 model for AWQ ...")
    model = AutoAWQForCausalLM.from_pretrained(args.model_dir, safetensors=True, device_map="auto")

    # AWQ API typically: model.quantize(tokenizer, quant_config=..., calib_data=[str,...])
    # We’ll pass raw text; AWQ will tokenize internally.
    # If your AWQ build needs tokenized, replace calib_data=texts with list of strings or token ids as needed.
    print("🧠 Running AWQ calibration+quantization ...")
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": args.bits,
            "version": "GEMM",
        },
        calib_data=texts[:args.calib_max_samples],
    )

    model.save_quantized(args.out_dir)
    print(f"✅ AWQ quantized model saved to: {args.out_dir}")

print("🎯 Quantization complete. Load with AutoGPTQForCausalLM.from_quantized(...) or AutoAWQForCausalLM.from_quantized(...).")
