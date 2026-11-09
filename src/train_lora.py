"""LoRA fine-tune of Qwen3-VL on chart -> table.

The model sees a chart and READ_PROMPT and learns to write the chart's true table
in the " | " format (tables.to_pipe). Only the answer tokens count in the loss.
LoRA goes on the language model's linear layers; the vision encoder stays frozen.

    python src/train_lora.py \
        --train data/synthetic_v2/train/manifest.jsonl data/chartqa_pairs/train/manifest.jsonl \
        --val   data/synthetic_v2/val/manifest.jsonl   data/chartqa_pairs/val/manifest.jsonl \
        --out outputs/lora/qwen4b_r16

Saves: adapter/ (best val loss), adapter_last/, log.jsonl, config.json.
Then: python src/merge_lora.py --adapter outputs/lora/qwen4b_r16/adapter --out ...
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from readers.prompts import READ_PROMPT
from tables import from_csv, to_pipe

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_examples(manifests, limit=None, seed=0):
    """[(image_path, target_text, source)] from one or more manifests."""
    out = []
    for m in manifests:
        m = Path(m)
        root = m.parent.parent
        recs = [json.loads(l) for l in m.open()]
        random.Random(seed).shuffle(recs)
        if limit:
            recs = recs[:limit]
        for r in recs:
            target = to_pipe(from_csv((root / r["table"]).read_text()))
            out.append((str(root / r["image"]), target, r.get("source", m.parent.parent.name)))
    return out


def open_image(path, max_side):
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


class Collator:
    """Builds input_ids / labels for a batch. Labels are -100 on the prompt, image and padding."""

    def __init__(self, processor, max_side):
        self.p = processor
        self.max_side = max_side
        self.p.tokenizer.padding_side = "right"

    def _messages(self, img, answer=None):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": READ_PROMPT}]}]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def __call__(self, batch):
        imgs = [open_image(path, self.max_side) for path, _, _ in batch]
        full = [self._messages(img, target) for img, (_, target, _) in zip(imgs, batch)]
        enc = self.p.apply_chat_template(full, tokenize=True, return_dict=True,
                                         return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, img in enumerate(imgs):
            prompt = self.p.apply_chat_template([self._messages(img)], tokenize=True, return_dict=True,
                                                add_generation_prompt=True, return_tensors="pt")
            labels[i, :prompt["input_ids"].shape[1]] = -100
        enc["labels"] = labels
        return enc


@torch.no_grad()
def val_loss(model, collate, examples, batch_size, device):
    model.eval()
    total, n = 0.0, 0
    by_source = {}
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        enc = {k: v.to(device) for k, v in collate(chunk).items()}
        loss = model(**enc).loss.item()
        total += loss * len(chunk)
        n += len(chunk)
        for _, _, src in chunk:
            by_source.setdefault(src, []).append(loss)
    model.train()
    return total / max(1, n), {s: sum(v) / len(v) for s, v in by_source.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--limit-train", type=int, default=None, help="max examples per train manifest")
    ap.add_argument("--limit-val", type=int, default=100, help="max examples per val manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke test)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-side", type=int, default=1024, help="resize charts so the longest side is at most this")
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps between val loss checks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log = (out / "log.jsonl").open("a")
    device = "cuda"

    train = load_examples(args.train, args.limit_train, args.seed)
    val = load_examples(args.val, args.limit_val, args.seed)
    random.shuffle(train)
    print(f"train {len(train)} examples, val {len(val)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    collate = Collator(processor, args.max_side)
    # sanity check: only the answer table should be in the loss
    check = collate(train[:1])
    kept = check["labels"][0][check["labels"][0] != -100]
    print("loss is computed on:", repr(processor.tokenizer.decode(kept)[:300]), flush=True)
    print("sequence length:", check["input_ids"].shape[1], flush=True)

    steps_per_epoch = math.ceil(len(train) / (args.batch * args.grad_accum))
    total_steps = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    warmup = max(1, int(0.03 * total_steps))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    def lr_at(step):  # linear warmup, then cosine to 10% of lr
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    v0, v0_src = val_loss(model, collate, val, args.batch, device)
    print(f"step 0  val_loss {v0:.4f}  {v0_src}", flush=True)
    log.write(json.dumps({"step": 0, "val_loss": v0, "val_by_source": v0_src}) + "\n")
    best = v0

    model.train()
    step, micro, t0, running = 0, 0, time.time(), []
    order = list(range(len(train)))
    while step < total_steps:
        random.shuffle(order)
        for i in range(0, len(order), args.batch):
            chunk = [train[j] for j in order[i:i + args.batch]]
            enc = {k: v.to(device) for k, v in collate(chunk).items()}
            loss = model(**enc).loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                rate = step / (time.time() - t0)
                eta = (total_steps - step) / max(rate, 1e-9) / 60
                tl = sum(running) / len(running)
                running = []
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"step {step}/{total_steps}  loss {tl:.4f}  lr {lr_at(step):.2e}  "
                      f"{rate * 60:.1f} steps/min  eta {eta:.0f} min  mem {mem:.1f} GB", flush=True)
                log.write(json.dumps({"step": step, "train_loss": tl, "lr": lr_at(step), "gpu_gb": mem}) + "\n")
                log.flush()

            if step % args.eval_every == 0 or step == total_steps:
                v, v_src = val_loss(model, collate, val, args.batch, device)
                print(f"step {step}  val_loss {v:.4f}  {v_src}", flush=True)
                log.write(json.dumps({"step": step, "val_loss": v, "val_by_source": v_src}) + "\n")
                log.flush()
                if v < best:
                    best = v
                    model.save_pretrained(out / "adapter")
                    print(f"  saved best adapter (val_loss {v:.4f})", flush=True)
            if step >= total_steps:
                break

    model.save_pretrained(out / "adapter_last")
    if not (out / "adapter").exists():
        model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val_loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
