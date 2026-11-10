"""Quick checks of the plain-Python parts. Run: python -m pytest tests"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checker import compare  # noqa: E402
from ensemble import combine  # noqa: E402
from metrics import detection_scores  # noqa: E402
from perturb import make_paper_table  # noqa: E402
from tables import Table, from_csv, parse_model_output, parse_number, to_pipe  # noqa: E402

TRUE = Table("Model", ["CIFAR-10", "ImageNet"], ["ResNet", "ViT", "BERT"],
             [[93.1, 76.2], [98.1, 81.0], [88.0, 70.5]])


def test_parse_number():
    assert parse_number("82.4%") == 82.4
    assert parse_number("1,234") == 1234
    assert parse_number("3,5") == 3.5
    assert parse_number("−2") == -2
    assert parse_number("n/a") is None


def test_parse_deplot_output():
    t = parse_model_output("TITLE | Acc <0x0A> Model | Accuracy <0x0A> ResNet | 82.4 <0x0A> ViT | 91")
    assert t.rows == ["ResNet", "ViT"] and t.values == [[82.4], [91.0]]


def test_parse_markdown_and_fences():
    t = parse_model_output("```\n| Model | A | B |\n|---|---|---|\n| X | 1 | 2 |\n```")
    assert t.series == ["A", "B"] and t.values == [[1.0, 2.0]]


def test_csv_roundtrip():
    assert from_csv(TRUE.to_csv()).cells() == TRUE.cells()


def test_match_and_mismatch():
    assert compare(TRUE, TRUE)["verdict"] == "match"
    bad = TRUE.copy()
    bad.values[1][1] = 60.0
    r = compare(TRUE, bad)
    assert r["verdict"] == "mismatch" and len(r["differences"]) == 1


def test_fuzzy_labels_and_series_by_order():
    chart = Table("x", ["series_1", "series_2"], ["Resnet", "VIT", "Bert"], TRUE.values)
    assert compare(chart, TRUE)["verdict"] == "match"


def test_unit_scale_is_found():
    paper = TRUE.copy()
    paper.values = [[v / 1000 for v in r] for r in paper.values]
    r = compare(TRUE, paper)
    assert r["verdict"] == "match" and r["scale"] == 1000


def test_adaptive_tolerance_and_not_sure():
    paper = TRUE.copy()
    paper.values[0][0] = 93.1 * 1.08                   # 8% off
    spread = {("ResNet", "CIFAR-10"): 0.05 * 93.1}     # but the reader was unsure (5%)
    assert compare(TRUE, paper, spread, tau=0.05, k=0)["verdict"] == "mismatch"
    assert compare(TRUE, paper, spread, tau=0.05, k=2)["verdict"] == "match"
    assert compare(TRUE, paper, spread, tau=0.05, k=0, s_max=0.03)["verdict"] == "not_sure"


def test_ensemble_median_and_spread():
    a = TRUE.copy()
    b = TRUE.copy(); b.values[0][0] = 95.1
    c = TRUE.copy(); c.values[0][0] = 91.1
    med, spread, info = combine([a, b, c, None])
    assert med.values[0][0] == 93.1 and spread[("ResNet", "CIFAR-10")] > 1 and info["n_ok"] == 3


def test_perturbations_are_real_mismatches():
    rng = random.Random(0)
    for _ in range(200):
        paper, info = make_paper_table(TRUE, rng, dec=1)
        if info["label"] == "match":
            assert compare(TRUE, paper)["verdict"] == "match"
        elif info["kind"] in ("swap", "decimal") or info["size"] >= 0.1:
            assert compare(TRUE, paper)["verdict"] == "mismatch", info


def test_detection_scores():
    s = detection_scores(["mismatch", "match", "mismatch", "match"],
                         ["mismatch", "mismatch", "not_sure", "match"])
    assert s["precision"] == 0.5 and s["recall"] == 1.0 and s["not_sure_rate"] == 0.25


def test_transposed_reading_is_turned_back():
    sideways = parse_model_output("City | Zurich | Geneva | Basel\n--- | --- | --- | ---\nPatients | 350 | 315 | 390")
    paper = Table("City", ["Patients"], ["Zurich", "Geneva", "Basel"], [[350.0], [315.0], [390.0]])
    assert compare(sideways, paper)["verdict"] == "match"
    paper.values[1][0] = 250.0
    assert compare(sideways, paper)["verdict"] == "mismatch"


def test_pipe_target_parses_back():
    assert parse_model_output(to_pipe(TRUE)).cells() == TRUE.cells()


def test_values_with_extra_text_are_kept():
    t = parse_model_output("Model | Acc\nA | 82.4 (±0.3)\nB | 75%")
    assert t.values == [[82.4], [75.0]]


def test_wide_output_without_series_name():
    t = parse_model_output("Year | 2014 | 2015 | 2016\n130 | 220 | 280")
    assert t.rows == ["2014", "2015", "2016"] and t.values == [[130.0], [220.0], [280.0]]
    paper = Table("Year", ["Revenue"], ["2014", "2015", "2016"], [[130.0], [220.0], [280.0]])
    assert compare(t, paper)["verdict"] == "match"


def test_wide_output_with_series_names():
    t = parse_model_output("City | Zurich | Geneva | Basel\nPlacebo | 1 | 2 | 3\nControl | 4 | 5 | 6")
    assert t.series == ["Placebo", "Control"] and t.values[2] == [3.0, 6.0]


def test_normal_long_table_is_not_taken_as_wide():
    t = parse_model_output("Model | A | B\nX | 1 | 2\nY | 3 | 4\nZ | 5 | 6")
    assert t.rows == ["X", "Y", "Z"] and t.series == ["A", "B"]


def test_arxiv_pgf_extracts_bar_and_line():
    from arxiv_pgf import extract_figures
    tex = r"""
\begin{tikzpicture}
\begin{axis}[ybar, symbolic x coords={ResNet,ViT,BERT}, xtick=data, ylabel={Accuracy (\%)}]
\addplot coordinates {(ResNet,93.1) (ViT,98.1) (BERT,88.0)}; \addlegendentry{CIFAR-10}
\addplot coordinates {(ResNet,76.2) (ViT,81.0) (BERT,70.5)}; \addlegendentry{ImageNet}
\end{axis}
\end{tikzpicture}
% a line chart with an inline table
\begin{tikzpicture}
\begin{axis}[xlabel={Epoch}]
\addplot table [x=ep, y=acc] {
ep acc
1 50.5
2 61.0
3 70.2
};
\end{axis}
\end{tikzpicture}
\begin{tikzpicture}\begin{axis}[only marks]\addplot+[only marks] coordinates {(1,2) (3,4)};\end{axis}\end{tikzpicture}
"""
    figs = list(extract_figures(tex, Path(".")))
    assert len(figs) == 2
    bar = from_csv(figs[0]["table"])
    assert figs[0]["chart_type"] == "v_bar" and bar.series == ["CIFAR-10", "ImageNet"]
    assert bar.cells()[("ViT", "ImageNet")] == 81.0
    line = from_csv(figs[1]["table"])
    assert figs[1]["chart_type"] == "line" and line.x_name == "Epoch" and line.values == [[50.5], [61.0], [70.2]]


def test_arxiv_pgf_skips_log_and_clipped_and_reads_legend():
    from arxiv_pgf import extract_figures
    log = r"\begin{tikzpicture}\begin{axis}[ymode=log]\addplot coordinates {(1,1) (2,0.1)};\end{axis}\end{tikzpicture}"
    clipped = r"\begin{tikzpicture}\begin{axis}[xmin=1]\addplot coordinates {(0,1) (1,2) (2,3)};\end{axis}\end{tikzpicture}"
    legend = (r"\begin{tikzpicture}\begin{axis}[ybar, symbolic x coords={T1,T2}]"
              r"\addplot coordinates {(T1,70) (T2,80)};\addplot coordinates {(T1,10) (T2,15)};"
              r"\legend{Old System, New System}\end{axis}\end{tikzpicture}")
    assert list(extract_figures(log, Path("."))) == []
    assert list(extract_figures(clipped, Path("."))) == []
    t = from_csv(list(extract_figures(legend, Path(".")))[0]["table"])
    assert t.series == ["Old System", "New System"]
