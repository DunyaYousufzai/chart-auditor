"""Merge a LoRA adapter into the base model and save a full model that vLLM can load.

    python src/merge_lora.py --adapter outputs/lora/qwen4b_r16/adapter --out outputs/lora/qwen4b_r16/merged

Then read charts with it exactly like the base model:
    python src/run_reader.py --reader qwen --model outputs/lora/qwen4b_r16/merged ...
"""
import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel

    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
