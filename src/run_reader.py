"""Run a chart reader on every chart of a manifest and save the raw texts.

    python src/run_reader.py --reader qwen --manifest data/synthetic_v2/val/manifest.jsonl \
        --n 5 --temperature 0.7 --out outputs/readings/qwen4b_val_n5.jsonl

We only save what the model wrote. Parsing, combining and checking happen in
evaluate.py, so we can try many checker settings without running the model again.
The output file can be resumed: ids already in it are skipped.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from readers import get_reader


def gpu_memory_used_gb():
    """GPU memory in use right now, from nvidia-smi (works for vLLM, whose model lives in another process).
    Note: vLLM reserves gpu_memory_utilization of the card up front, so this shows the reservation."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split()) / 1024
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=["deplot", "qwen", "qwen-hf", "simulated"])
    ap.add_argument("--model", default=None, help="Hugging Face id or local path")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None, help="folder the image paths are relative to (default: manifest/../..)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="readings per chart")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--task", default="read", choices=["read", "judge"])
    ap.add_argument("--batch", type=int, default=64, help="charts sent to the model at once")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=None, help="shrink images to this many pixels (speed test)")
    ap.add_argument("--quantization", default=None, help="vLLM quantization, e.g. fp8")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    root = Path(args.root) if args.root else manifest.parent.parent
    records = [json.loads(l) for l in manifest.open()]
    if args.limit:
        records = records[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open()}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} charts, {len(done)} already done, {len(todo)} to read")
    if not todo:
        return

    kw = {}
    if args.model:
        kw["model"] = args.model
    if args.max_side:
        kw["max_side"] = args.max_side
    if args.quantization:
        kw["quantization"] = args.quantization
    t0 = time.time()
    reader = get_reader(args.reader, **kw)
    print(f"model loaded in {time.time() - t0:.0f}s: {reader.name}")

    peak_gb = gpu_memory_used_gb()
    extra = {"task": args.task} if args.reader.startswith("qwen") else {}
    with out.open("a") as f:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            paths = [str(root / r["image"]) for r in batch]
            if args.task == "judge":
                extra["tables"] = [(root / r["paper_table"]).read_text() for r in batch]
            t = time.time()
            texts = reader.read(paths, n=args.n, temperature=args.temperature, seed=args.seed, **extra)
            per_chart = (time.time() - t) / len(batch)
            mem = gpu_memory_used_gb()
            if mem is not None:
                peak_gb = max(peak_gb or 0, mem)
            for r, outs in zip(batch, texts):
                f.write(json.dumps({"id": r["id"], "reader": args.reader, "model": reader.name,
                                    "task": args.task, "n": args.n, "temperature": args.temperature,
                                    "max_side": args.max_side, "quantization": args.quantization,
                                    "seconds_per_chart": round(per_chart, 3),
                                    "gpu_gb": None if peak_gb is None else round(peak_gb, 2),
                                    "outputs": outs}) + "\n")
            f.flush()
            print(f"{start + len(batch)}/{len(todo)}  {per_chart:.2f}s per chart")

    if peak_gb is not None:
        print(f"peak GPU memory (nvidia-smi): {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
