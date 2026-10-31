"""run_tests.py — full suite runner used by the plan's ledger task-done calls."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.exit(subprocess.run([str(root / ".venv" / "bin" / "pytest"), "tests", "-q"],
                            cwd=root).returncode)
