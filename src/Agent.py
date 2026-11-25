from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from checker import compare
from ensemble import combine
from Notation_Normalizer_Agent import NotationNormalizer, TableNormReport
from Resotuion_Zoom_Agent import ResolutionZoomAgent, ZoomResult
from tables import Table


@dataclass
class AgentTrace:
    actions: List[str] = field(default_factory=list)
    zoom_results: List[ZoomResult] = field(default_factory=list)
    extra_readings: int = 0
    ambiguous_cells_found: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class ChartAuditorAgent:
    """Orquesta inspección/zoom de imagen, normalización numérica, ensamble y comparación."""

    def __init__(
        self,
        read_fn: Callable[[str, int], List[Table]],
        normalizer: Optional[NotationNormalizer] = None,
        zoom_agent: Optional[ResolutionZoomAgent] = None,
        spread_threshold: float = 0.08,
        tau: float = 0.05,
        k: float = 1.5,
        decimal_hint: Optional[str] = None,
    ):
        self.read_fn = read_fn
        self.normalizer = normalizer or NotationNormalizer()
        self.zoom_agent = zoom_agent or ResolutionZoomAgent()
        self.spread_threshold = spread_threshold
        self.tau = tau
        self.k = k
        self.decimal_hint = decimal_hint

    def _has_high_spread(self, spread: Dict[Tuple[str, str], float], table: Table) -> bool:
        if not spread:
            return False
        cells = table.cells()
        for key, sp in spread.items():
            ref = abs(cells.get(key, 0.0))
            rel = sp / max(ref, 1e-9)
            if rel > self.spread_threshold:
                return True
        return False

    def _normalize_readings(self, readings: List[Table]) -> Tuple[List[Table], List[TableNormReport]]:
        norm_tables: List[Table] = []
        reports: List[TableNormReport] = []
        for r in readings:
            if r is None:
                continue
            rep = self.normalizer.normalize_table(r, decimal_hint=self.decimal_hint)
            reports.append(rep)
            norm_tables.append(rep.table)
        return norm_tables, reports

    def audit(
        self,
        image_path: str,
        paper_table: Table,
        n_initial: int = 3,
        n_retry: int = 3,
    ) -> Tuple[dict, AgentTrace]:
        trace = AgentTrace()
        active_image = image_path

        # 1. Revisión inicial de resolución/contraste de la imagen
        should_z, reason, _ = self.zoom_agent.should_zoom(active_image)
        if should_z:
            z_res = self.zoom_agent.zoom_image(active_image, reason=reason, tag="init_zoom")
            trace.zoom_results.append(z_res)
            if z_res.applied:
                active_image = z_res.image_path
                trace.actions.append(f"zoom_init({reason})")

        # 2. Lectura inicial + normalización de notaciones/comas/unidades
        raw_readings = self.read_fn(active_image, n_initial)
        norm_tables, norm_reports = self._normalize_readings(raw_readings)
        combined, spread, _ = combine(norm_tables)

        ambiguous = []
        for rep in norm_reports:
            ambiguous.extend(rep.ambiguous_cells)
        trace.ambiguous_cells_found = list(dict.fromkeys(ambiguous))

        # 3. Decisión de reintento: si no se pudo combinar, hay dispersión alta o celdas ambiguas
        high_sp = self._has_high_spread(spread, combined) if combined is not None else True
        has_amb = len(trace.ambiguous_cells_found) > 0

        if combined is None or high_sp or has_amb:
            # Si aún no hicimos zoom sobre la imagen original, intentamos ampliar antes de releer
            already_upscaled = any(zr.applied for zr in trace.zoom_results)
            if not already_upscaled:
                retry_reason = "high_spread" if high_sp else ("ambiguous_notation" if has_amb else "empty_read")
                z_retry = self.zoom_agent.zoom_image(image_path, reason=retry_reason, tag="retry_zoom")
                trace.zoom_results.append(z_retry)
                if z_retry.applied:
                    active_image = z_retry.image_path
                    trace.actions.append(f"zoom_retry({retry_reason})")

            extra_raw = self.read_fn(active_image, n_retry)
            trace.extra_readings += len(extra_raw)
            trace.actions.append(f"retry_read(n={n_retry})")

            extra_norm, extra_reports = self._normalize_readings(extra_raw)
            all_tables = norm_tables + extra_norm
            combined, spread, _ = combine(all_tables)

            for rep in extra_reports:
                for cell in rep.ambiguous_cells:
                    if cell not in trace.ambiguous_cells_found:
                        trace.ambiguous_cells_found.append(cell)

        # 4. Normalizar tabla de referencia y comparar con tolerancia/dispersión en Python
        paper_report = self.normalizer.normalize_table(paper_table, decimal_hint=self.decimal_hint)
        if combined is None:
            verdict = {
                "verdict": "not_sure",
                "reason": "could_not_parse_chart_table",
                "differences": [],
            }
        else:
            verdict = compare(
                combined,
                paper_report.table,
                spread=spread,
                tau=self.tau,
                k=self.k,
            )

        return verdict, trace