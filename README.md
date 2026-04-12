# AML Project 1

This repository contains code for Project 1 (missing labels in binary logistic regression), including:

- data loading and missing-label generation schemes (`dataset_prep.py`),
- FISTA solver for logistic Lasso (`FISTA.py`),
- labeled logistic regression wrapper with lambda validation (`labeled_log_reg.py`),
- missing-label handling wrapper with Naive/Oracle/LDA/KNN strategies (`unlabeled_log_reg.py`),
- experiment notebook (`AMLpro1.ipynb`),
- report (`reports/AMLreport1.tex`).

## Environment setup

1. Install uv (if not installed):

macOS:

```bash
brew install uv
```

Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
winget install --id=astral-sh.uv -e
```

1. Create and sync the environment with uv:

```bash
uv venv
uv sync
```

1. Activate the environment:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
.venv\Scripts\activate.bat
```

1. If you need to add packages manually with uv:

```bash
uv add numpy pandas scikit-learn matplotlib kagglehub jupyter
```

Optional fallback (without uv):

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib kagglehub jupyter
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install numpy pandas scikit-learn matplotlib kagglehub jupyter
```

## Running experiments

Main workflow is in the notebook:

```bash
uv run jupyter notebook AMLpro1.ipynb
```

Windows (equivalent command):

```powershell
uv run jupyter notebook AMLpro1.ipynb
```

Run cells in order to:

- prepare datasets,
- generate missing labels (MCAR, MAR1, MAR2, MNAR),
- compare FISTA vs `sklearn` logistic regression,
- evaluate Naive/Oracle/LDA/KNN approaches.

## Running experiments in VS Code

1. Open the project folder in VS Code.
1. Install extensions (if needed):
   - Python (`ms-python.python`)
   - Jupyter (`ms-toolsai.jupyter`)
1. Open `AMLpro1.ipynb`.
1. Select kernel from `.venv` (top-right kernel selector in notebook editor).
1. Run notebook cells from top to bottom.

Using uv from VS Code terminal:

macOS/Linux:

```bash
source .venv/bin/activate
uv run jupyter notebook AMLpro1.ipynb
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
uv run jupyter notebook AMLpro1.ipynb
```

If VS Code asks to install `ipykernel` for the selected environment, confirm the prompt.

## Using the modules in scripts

Example skeleton:

```python
from dataset_prep import get_dataset
from labeled_log_reg import LabeledLogReg
from unlabeled_log_reg import UnlabeledLogReg

train_df, val_df, test_df = get_dataset("wine")

target = "quality"
X_train = train_df.drop(columns=[target]).to_numpy()
y_train_obs = train_df[f"{target}_mcar"].to_numpy()  # contains -1 for missing labels

X_val = val_df.drop(columns=[target]).to_numpy()
y_val = val_df[target].to_numpy()

base_model = LabeledLogReg(implementation="fista")
ulr = UnlabeledLogReg()
fitted = ulr.fit(base_model, X_train, y_train_obs, method="knn")
fitted.validate(X_val, y_val, measure="f1")
```

## Notes

- Data download uses `kagglehub`, so internet access is required.
- Missing labels are represented as `-1` in generated training targets.
- Random seed is fixed in dataset preparation (`RANDOM_SEED = 42`) for reproducibility.
