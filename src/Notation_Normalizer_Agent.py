from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple

from tables import Table, parse_number


@dataclass
class CellNormInfo:
    raw: str
    value: Optional[float]
    status: str  # "exact", "normalized", "scaled", "ambiguous", "qualifier", "missing"
    qualifier: Optional[str] = None  # e.g. "<", ">", "~", "range", "na"
    scale_applied: float = 1.0
    note: str = ""


@dataclass
class TableNormReport:
    table: Table
    decimal_mode: str  # "dot", "comma", "mixed_or_unknown"
    context_scale: float
    cells: Dict[Tuple[str, str], CellNormInfo] = field(default_factory=dict)

    @property
    def ambiguous_cells(self) -> List[Tuple[str, str]]:
        return [k for k, info in self.cells.items() if info.status in ("ambiguous", "qualifier")]


class NotationNormalizer:
    """Normaliza notaciones numéricas usando el contexto de toda la tabla y registra trazas."""

    SUFFIX_MULTIPLIERS = {
        "k": 1e3,
        "thousand": 1e3,
        "thousands": 1e3,
        "m": 1e6,
        "million": 1e6,
        "millions": 1e6,
        "b": 1e9,
        "billion": 1e9,
        "billions": 1e9,
    }

    MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "—", "–", "missing"}

    def _infer_context_scale(self, table: Table) -> float:
        context_text = " ".join(
            [table.x_name or ""] + list(table.series or []) + list(table.rows or [])
        ).lower()

        if re.search(r"\b(billion|billions|\$?b)\b|(\(\s*b\s*\)|\[\s*b\s*\])", context_text):
            return 1e9
        if re.search(r"\b(million|millions|\$?m)\b|(\(\s*m\s*\)|\[\s*m\s*\])", context_text):
            return 1e6
        if re.search(r"\b(thousand|thousands|\$?k)\b|(\(\s*k\s*\)|\[\s*k\s*\])", context_text):
            return 1e3
        return 1.0

    def _infer_decimal_mode(self, table: Table, decimal_hint: Optional[str] = None) -> str:
        if decimal_hint in (",", "comma"):
            return "comma"
        if decimal_hint in (".", "dot"):
            return "dot"

        comma_as_decimal = 0
        comma_as_thousands = 0

        for row in table.values:
            for val in row:
                if val is None or isinstance(val, (int, float)):
                    continue
                s = str(val).strip()
                # Formato europeo claro: 1.234,56
                if re.search(r"\d+\.\d{3},\d+", s):
                    comma_as_decimal += 2
                # Formato anglosajón claro: 1,234.56
                elif re.search(r"\d+,\d{3}\.\d+", s):
                    comma_as_thousands += 2
                # Una coma seguida de 1 o 2 dígitos -> decimal muy probable (ej. 3,5 o 12,45)
                elif re.search(r"(?<!\d)\d+,\d{1,2}(?!\d)", s):
                    comma_as_decimal += 1
                # Una coma seguida de exactamente 3 dígitos -> puede ser miles (1,250)
                elif re.search(r"(?<!\d)\d+,\d{3}(?!\d)", s):
                    comma_as_thousands += 1

        if comma_as_decimal > comma_as_thousands and comma_as_decimal > 0:
            return "comma"
        if comma_as_thousands > comma_as_decimal and comma_as_thousands > 0:
            return "dot"
        return "mixed_or_unknown"

    def normalize_cell(
        self,
        raw: object,
        decimal_mode: str = "mixed_or_unknown",
        context_scale: float = 1.0,
    ) -> CellNormInfo:
        if raw is None:
            return CellNormInfo(raw="", value=None, status="missing", qualifier="na", note="None value")

        if isinstance(raw, (int, float)):
            val = float(raw)
            scaled = val * context_scale
            status = "scaled" if context_scale != 1.0 else "exact"
            return CellNormInfo(
                raw=str(raw),
                value=scaled,
                status=status,
                scale_applied=context_scale,
                note="numeric" if context_scale == 1.0 else f"context_scale={context_scale}",
            )

        s = str(raw).strip()
        low = s.lower()
        if low in self.MISSING_TOKENS:
            return CellNormInfo(raw=s, value=None, status="missing", qualifier="na", note="missing token")

        # Rangos tipo "10-12" o "10–12" (sin signo menos inicial)
        if re.match(r"^\s*\d+(?:[.,]\d+)?\s*[–—-]\s*\d+(?:[.,]\d+)?\s*$", s):
            return CellNormInfo(raw=s, value=None, status="qualifier", qualifier="range", note="interval/range")

        # Calificadores de desigualdad o aproximación: <0.01, >5, ~10, ≈20
        qualifier = None
        work = s.replace("\u2212", "-").replace("−", "-")
        m_qual = re.match(r"^\s*([<>≈~])\s*(.+)$", work)
        if m_qual:
            q_sym, rest = m_qual.group(1), m_qual.group(2).strip()
            qualifier = "~" if q_sym in ("~", "≈") else q_sym
            work = rest

        # Sufijo explícito de magnitud (k, m, b, etc.)
        suffix_mult = 1.0
        m_suf = re.search(
            r"(?i)(?:^|[\s$€£])(-?\d+(?:[.,]\d+)?)\s*(k|thousand|thousands|m|million|millions|b|billion|billions)\s*%?$",
            work,
        )
        if m_suf:
            num_part, suf = m_suf.group(1), m_suf.group(2).lower()
            suffix_mult = self.SUFFIX_MULTIPLIERS[suf]
            work = num_part

        # Intento directo con parse_number existente
        direct = parse_number(work)
        if direct is not None and "," not in work:
            total_mult = suffix_mult * (context_scale if suffix_mult == 1.0 else 1.0)
            final_val = direct * total_mult
            if qualifier is not None:
                return CellNormInfo(
                    raw=s,
                    value=final_val,
                    status="qualifier",
                    qualifier=qualifier,
                    scale_applied=total_mult,
                    note=f"qualifier={qualifier}",
                )
            status = "scaled" if total_mult != 1.0 else "exact"
            return CellNormInfo(
                raw=s,
                value=final_val,
                status=status,
                scale_applied=total_mult,
                note="direct" if total_mult == 1.0 else f"mult={total_mult}",
            )

        # Limpieza para resolver comas/puntos
        cleaned = re.sub(r"[^\d,.\-eE+]", "", work)
        status_note = "comma_resolved"
        ambiguous = False

        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
                status_note = "eu_dot_thousands_comma_dec"
            else:
                cleaned = cleaned.replace(",", "")
                status_note = "us_comma_thousands_dot_dec"
        elif "," in cleaned:
            parts = cleaned.split(",")
            if decimal_mode == "comma":
                cleaned = cleaned.replace(",", ".")
                status_note = "context_comma_decimal"
            elif decimal_mode == "dot":
                cleaned = cleaned.replace(",", "")
                status_note = "context_comma_thousands"
            else:
                # Sin contexto claro: si son 3 dígitos tras la coma, es ambiguo entre miles y decimal
                if len(parts) == 2 and len(parts[1]) == 3 and parts[0].lstrip("-").isdigit():
                    cleaned = cleaned.replace(",", "")
                    ambiguous = True
                    status_note = "ambiguous_3digits_after_comma_assumed_thousands"
                elif len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
                    cleaned = cleaned.replace(",", "")
                    status_note = "multi_comma_thousands"
                else:
                    cleaned = cleaned.replace(",", ".")
                    status_note = "fallback_comma_to_dot"

        parsed = parse_number(cleaned)
        if parsed is None:
            return CellNormInfo(
                raw=s,
                value=None,
                status="ambiguous",
                qualifier=qualifier,
                note=f"unparsed_after_clean({cleaned})",
            )

        total_mult = suffix_mult * (context_scale if suffix_mult == 1.0 else 1.0)
        final_val = parsed * total_mult

        if qualifier is not None:
            return CellNormInfo(
                raw=s,
                value=final_val,
                status="qualifier",
                qualifier=qualifier,
                scale_applied=total_mult,
                note=f"{status_note}; qualifier={qualifier}",
            )

        if ambiguous:
            return CellNormInfo(
                raw=s,
                value=final_val,
                status="ambiguous",
                scale_applied=total_mult,
                note=status_note,
            )

        final_status = "scaled" if total_mult != 1.0 else "normalized"
        return CellNormInfo(
            raw=s,
            value=final_val,
            status=final_status,
            scale_applied=total_mult,
            note=status_note,
        )

    def normalize_table(self, table: Table, decimal_hint: Optional[str] = None) -> TableNormReport:
        dec_mode = self._infer_decimal_mode(table, decimal_hint=decimal_hint)
        ctx_scale = self._infer_context_scale(table)
        report = TableNormReport(
            table=Table(table.x_name, list(table.series), list(table.rows), []),
            decimal_mode=dec_mode,
            context_scale=ctx_scale,
        )

        new_values: List[List[Optional[float]]] = []
        for r_idx, row in enumerate(table.values):
            row_label = table.rows[r_idx] if r_idx < len(table.rows) else str(r_idx)
            new_row: List[Optional[float]] = []
            for c_idx, raw_val in enumerate(row):
                s_name = table.series[c_idx] if c_idx < len(table.series) else str(c_idx)
                info = self.normalize_cell(raw_val, decimal_mode=dec_mode, context_scale=ctx_scale)
                report.cells[(row_label, s_name)] = info
                new_row.append(info.value)
            new_values.append(new_row)

        report.table.values = new_values
        return report