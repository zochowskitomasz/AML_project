"""Thin entry point for the Project 2 Logistic Regression experiment.

The full experiment logic lives in `logistic_regression_business_experiment.py`.
This file exists so teammates can run `python logistic_regression_experiment.py`
without hunting for the business-scoring script.
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from logistic_regression_business_experiment import run_experiment  # noqa: E402


def main() -> None:
    summary = run_experiment(fast=True)
    print(f"best_k={summary['best_k']}")
    print(f"best_cv_score={summary['best_cv_business_score']}")
    print(f"n_variables={summary['n_variables']}")
    print(f"output_dir={summary['output_dir']}")


if __name__ == "__main__":
    main()
