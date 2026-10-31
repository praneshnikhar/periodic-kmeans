"""validate_nb.py — notebook QA gate: JSON valid + every code cell parses.

Test command used by the plan's tasks (one arg = notebook path).
"""
import ast
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_nb.py <notebook.ipynb>")
        return 2
    nbp = Path(sys.argv[1])
    nb = json.loads(nbp.read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                print(f"FAIL {nbp.name} cell {i}: {e}")
                return 1
    print(f"AST-VALIDATE {nbp.name}: PASS (all {sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells parse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
