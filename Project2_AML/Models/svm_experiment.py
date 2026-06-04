"""Thin entry point for the Project 2 SVM experiment.

The full experiment logic lives in `svm_business_experiment.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from svm_business_experiment import run_experiment  # noqa: E402


def main() -> None:
    summary = run_experiment()
    print(f"best_k={summary['best_k']}")
    print(f"best_cv_score={summary['best_cv_business_score']}")
    print(f"n_variables={summary['n_variables']}")
    print(f"output_dir={summary['output_dir']}")


if __name__ == "__main__":
    main()
