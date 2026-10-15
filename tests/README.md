# 🧪 Tests

This folder contains the automated tests for the House Price Prediction project.
The tests follow the **3 dimensions of CI for Machine Learning**: Code, Data, Model.

| File | CI Layer | What it protects |
|------|----------|------------------|
| `code_test.py` | Code CI | Logic of the data split function |
| `data_test.py` | Data CI | Schema and value bounds of the dataset |
| `model_test.py` | Model CI | Minimum model performance (performance gate) |

---

## ▶️ How to Run

Run from the **project root** (not from inside `tests/`):

```bash
# everything
pytest tests/ -v

# a single file
pytest tests/code_test.py
pytest tests/data_test.py
pytest tests/model_test.py
```

Install the dependencies first:

```bash
pip install -r requirements.txt
```

> Each test file adds the parent folder to `sys.path`, so `from train import ...` works without installing the project as a package.

---

## 1️⃣ `code_test.py` — Code CI (Unit Test)

**Function tested:** `split_my_data()` from `train.py`

| Test | Checks |
|------|--------|
| `test_data_split_ratio` | Training set is ~80% of the data (tolerance of 1 row) |
| | `len(X_train) == len(y_train)` (features and target are aligned) |

**Why:** if someone changes the split logic by mistake (for example 70/30), this test fails immediately.

---

## 2️⃣ `data_test.py` — Data CI

**Function tested:** `load_real_data()` from `train.py`

| Check | Type |
|-------|------|
| No NULL values in `X` | Data quality |
| No NULL values in `y` | Data quality |
| `y > 0` (price) | Value bound |
| `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population` all `> 0` | Value bound |
| All 8 expected columns exist | Schema check |

**Expected columns:**
`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

**Why:** a house price can never be negative, and the model needs all its input columns. If the data source changes or breaks, this test catches it before training.

---

## 3️⃣ `model_test.py` — Model CI (Performance Gate)

**Function tested:** `train_and_evaluate()` from `train.py`

The model is trained with the best hyperparameters found in the MLflow experiments (**Run 2**: `max_depth=5`, `learning_rate=0.05`) and must pass:

| Metric | Condition |
|--------|-----------|
| RMSE | `< 0.53` |
| R² | `> 0.79` |

**Why:** a model that does not meet the minimum quality must never be pushed to production.

> ⚠️ If you change the model, the data, or the best hyperparameters, update the thresholds (and the values in this file) based on the new best MLflow run.

---

## 🔄 When Do the Tests Run in CI?

| Test | Push to `feature/**` | Push to `main` | Pull Request to `main` |
|------|:--------------------:|:--------------:|:----------------------:|
| `code_test.py` | ✅ | ✅ | ✅ |
| `data_test.py` | ❌ | ✅ | ✅ |
| `model_test.py` | ❌ | ✅ | ✅ |

The data and model tests are heavier (they download data and train a model), so they only run before merging into `main`.

Workflow file: [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)

---

## ➕ Adding a New Test

1. Create a file in `tests/` that ends with `_test.py` (or `test_*.py`).
2. Add the same `sys.path` lines at the top, then import from `train.py`.
3. Write functions that start with `test_`.
4. Add a step for it in `ci.yml` if you want to control when it runs.

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import load_real_data


def test_example():
    x, y = load_real_data()
    assert len(x) == len(y)
```
