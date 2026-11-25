import argparse
import json
from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from Agent import ChartAuditorAgent
from Notation_Normalizer_Agent import NotationNormalizer
from readers import get_reader
from Resotuion_Zoom_Agent import ResolutionZoomAgent
from tables import Table, from_csv, parse_model_output


def _table_from_manifest_row(row: dict, root: Path) -> Table:
    """Carga la tabla de referencia ('paper_table' o 'table') desde el manifest."""
    for key in ("paper_table", "table", "ground_truth_table"):
        rel = row.get(key)
        if rel:
            p = root / rel
            if p.exists():
                return from_csv(p.read_text())
    raise KeyError(f"No encontré ruta de tabla en el registro: {list(row.keys())}")


def main():
    ap = argparse.ArgumentParser(description="Prueba ChartAuditorAgent sobre un manifest real")
    ap.add_argument("--manifest", required=True, help="Ruta a manifest.jsonl")
    ap.add_argument("--root", default=None, help="Raíz de rutas relativas (default: manifest/../..)")
    ap.add_argument("--reader", default="simulated", choices=["deplot", "qwen", "qwen-hf", "simulated"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--n-initial", type=int, default=3)
    ap.add_argument("--n-retry", type=int, default=2)
    ap.add_argument("--min-side", type=int, default=1000)
    ap.add_argument("--decimal-hint", default=None, choices=[".", ","])
    ap.add_argument("--out", default="outputs/results/agent_pilot_sample.jsonl")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(args.root) if args.root else manifest_path.parent.parent
    records = [json.loads(line) for line in manifest_path.open() if line.strip()][: args.limit]

    reader_kw = {"model": args.model} if args.model else {}
    reader = get_reader(args.reader, **reader_kw)

    # Mapea imágenes con zoom a la imagen original para que SimulatedReader encuentre su CSV
    zoom_to_orig = {}
    current_orig_img = None

    def read_fn(img_path: str, n: int):
        if args.reader == "simulated":
            reader_img = zoom_to_orig.get(img_path, current_orig_img or img_path)
        else:
            reader_img = img_path

        raw_batches = reader.read([reader_img], n=n, temperature=0.7 if n > 1 else 0.0)
        raw_texts = raw_batches[0] if raw_batches else []
        parsed = []
        for txt in raw_texts:
            try:
                parsed.append(parse_model_output(txt))
            except Exception:
                parsed.append(None)
        return parsed

    zoom_agent = ResolutionZoomAgent(min_side_px=args.min_side)
    normalizer = NotationNormalizer()
    agent = ChartAuditorAgent(
        read_fn=read_fn,
        normalizer=normalizer,
        zoom_agent=zoom_agent,
        decimal_hint=args.decimal_hint,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zoom_count = 0
    retry_count = 0
    ambiguous_count = 0

    with out_path.open("w") as fout:
        for row in records:
            img_path = str(root / row["image"])
            current_orig_img = img_path
            paper_tbl = _table_from_manifest_row(row, root)

            verdict, trace = agent.audit(
                img_path,
                paper_tbl,
                n_initial=args.n_initial,
                n_retry=args.n_retry,
            )

            for zr in trace.zoom_results:
                if zr.applied and zr.image_path:
                    zoom_to_orig[zr.image_path] = img_path

            used_zoom = any(z.applied for z in trace.zoom_results)
            used_retry = trace.extra_readings > 0
            has_amb = len(trace.ambiguous_cells_found) > 0

            zoom_count += int(used_zoom)
            retry_count += int(used_retry)
            ambiguous_count += int(has_amb)

            item = {
                "id": row.get("id"),
                "expected": row.get("label") or row.get("verdict"),
                "predicted": verdict.get("verdict"),
                "used_zoom": used_zoom,
                "extra_readings": trace.extra_readings,
                "ambiguous_cells": [list(c) for c in trace.ambiguous_cells_found],
                "actions": trace.actions,
            }
            fout.write(json.dumps(item) + "\n")
            print(
                f"[{row.get('id')}] pred={item['predicted']} | "
                f"zoom={used_zoom} | retry={trace.extra_readings} | "
                f"amb={len(trace.ambiguous_cells_found)} | actions={trace.actions}"
            )

    print(
        f"\nResumen ({len(records)} casos): "
        f"con_zoom={zoom_count}, con_relectura={retry_count}, con_ambiguedad={ambiguous_count}"
    )
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()