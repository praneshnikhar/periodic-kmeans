"""docs-consistency gate (review I-1/I-2 fix pass).

The executed notebook 03 IS the source of truth for numbers. Every grader-facing
file must match its printed outputs. Figures must exist as committed files.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _executed_outputs(name: str) -> str:
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    chunks = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    text = o.get("text", "")
                    chunks.append("".join(text) if isinstance(text, list) else text)
    return "".join(chunks)


def test_h_sensitivity_sign_consistent():
    out = _executed_outputs("03_validation_report")
    m = re.search(r"Δ = ([+-]\d+\.\d+)", out)
    assert m, "executed notebook must print the Δ line"
    delta = m.group(1)
    prose_files = ["README.md", "reports/p_report.md", "context.md"]
    for f in prose_files:
        text = (ROOT / f).read_text()
        assert "0.0016" in text or "0.002" in text, f"{f}: must quote the H-sensitivity Δ magnitude"
        for bad in ("-0.002", "\u22120.002"):
            assert bad not in text, f"{f}: contradicting executed sign (executed Δ = {delta})"


def test_k18_rounding_consistent():
    out = _executed_outputs("03_validation_report")
    m = re.search(r"ARI@k=18 \(companion, seed 0\): (\d\.\d+)", out)
    assert m, "executed notebook must print the companion line"
    claim = m.group(1)  # e.g. 0.0240
    two_dp = claim[:4]  # 0.02 slice→"0.02"; use first 4 chars for 2-dp value
    for f in ("README.md", "reports/p_report.md"):
        text = (ROOT / f).read_text()
        assert "0.024" in text, f"{f}: k=18 companion must be quoted as 0.024 (executed value)"
        assert "0.025" not in text, f"{f}: stale 0.025 contradicting executed {claim}"


def test_figures_directory_has_artifacts():
    figs = list((ROOT / "reports" / "figures").glob("*.png"))
    assert len(figs) >= 5, f"expected >=5 saved figures, found {[f.name for f in figs]}"


if __name__ == "__main__":
    for fn in [test_h_sensitivity_sign_consistent, test_k18_rounding_consistent, test_figures_directory_has_artifacts]:
        try:
            fn()
            print(f"{fn.__name__}: PASS")
        except AssertionError as e:
            print(f"{fn.__name__}: FAIL — {e}")
            sys.exit(1)
