#!/usr/bin/env python3
"""
06_export_adapters.py
Merge a LLaMA 3.2-1B Instruct base model with its QLoRA adapter (qlora-judge-ckpt)
to produce a single FP16 merged model.
"""

import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model ID or path, e.g. 'meta-llama/Llama-3.2-1B-Instruct'")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter folder, e.g. './qlora-judge-ckpt'")
    ap.add_argument("--out", required=True, help="Output folder for merged FP16 model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"🔹 Loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto"
    )

    print(f"🔹 Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("🔹 Merging adapter weights into base model…")
    merged = model.merge_and_unload()

    print(f"🔹 Loading tokenizer for: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"💾 Saving merged model to: {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    print("\n✅ Merge complete! Merged FP16 model saved to:", args.out)

if __name__ == "__main__":
    main()
