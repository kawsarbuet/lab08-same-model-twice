# Lakeshore Bike Demand: the same model in PyTorch and Keras

## What it does

AIGC 5005 Lab 08 compares the **original Lab 07 PyTorch model** with the same network in **Keras 3 (TensorFlow backend)**. The supplied simulated data and `bikes_data.py` are unchanged. There is **no deployment**. Original Lab 07 project files, tests, and reports are retained.

## How to run it

From this project folder, using **Python 3.12** and Conda:

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
python -c "import torch, tensorflow, keras; print(torch.__version__, tensorflow.__version__, keras.__version__)"
python train.py
python train_keras.py
python compare.py
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The three scripts recreate `reports/`. `train.py` is Lab 07's original PyTorch experiments; `train_keras.py` checks equivalence and creates the **main Keras run**; `compare.py` executes six full training runs (three seeds x two frameworks). Every test split is scored once per trained final model. Run from a **fresh clone** before submission. Installation requires internet and significant disk space.

## Data and split

`bikes_data.py` generates 6,000 synthetic rows from seed 2026. `prepare.py` divides rows 60/20/20 into train, validation, and test, and fits `StandardScaler` and `OneHotEncoder` only on training. Both training scripts use the **same** `Prepared` arrays for each seed in `compare.py` (2026, 2027, 2028). There is no independent Keras preprocessing pipeline. Eight input features are float32.

## Model and training

Both models are Dense/Linear(8 -> 64), ReLU, Dense/Linear(64 -> 64), ReLU, Dense/Linear(64 -> 1), with no output activation. The main configuration uses MSE, Adam, lr 0.003, batch 128, 100 epochs. PyTorch restores best validation-MSE weights manually; Keras uses `EarlyStopping(patience=epochs, restore_best_weights=True)` and `ModelCheckpoint(save_best_only=True)` to do the same. Note: initial weights and shuffle sequences may differ between frameworks despite common seeds. Validate on exactly the same rows and flatten Keras `(N,1)` predictions before MAE.

Keras weights are transferred **from an untrained PyTorch model** in `keras_model.py`: PyTorch `(outputs, inputs)` weights must be transposed to Keras `(inputs, outputs)`. `train_keras.py` prints the maximum validation prediction error, asserting <=1e-5. The transferred network is **only used to prove function equivalence**, not to initialize Keras's separate fairness training run.

## Results

The *original Lab 07* reported a validation-selected **wide-network** model test MAE of **2.414**; that is not the same model configuration as the Lab 08 main comparison and must not be represented as its result. Lab 07 curves and statistics remain in `reports/`.

**Lab 08 results:** `reports/comparison.csv` is produced when `python compare.py` successfully completes in your TensorFlow environment. The file should contain 6 rows with framework, seed, best epoch, validation MAE, test MAE and seconds. **Fill the comparison summary below only after the TensorFlow run:**

| Framework | Mean test MAE across 3 seeds | Mean seconds |
|---|---:|---:|
| PyTorch | PENDING TensorFlow-environment comparison | PENDING |
| Keras (TensorFlow) | PENDING | PENDING |

The included `reports/learning_curves_keras.png` and `reports/history_keras.csv` were produced locally using **Keras 3 with its PyTorch backend as a preliminary smoke check**, because TensorFlow could not be installed in this offline workspace. They are **not TensorFlow comparison results**. Recreate and replace them by running `python train_keras.py` in your required TensorFlow environment. A local preview achieved best epoch 94, validation MAE 2.5341, and test MAE 2.4634 — these are provisional, not a cross-framework benchmark.

## Experiments

The historical Lab 07 experiment artifacts in `reports/` include main, low-learning-rate and wide-network runs, plus linear and mean baselines. For Lab 08, `compare.py` uses **only the main 64-wide architecture** across seeds 2026, 2027 and 2028 in each framework. Each framework trains 100 epochs per seed; Keras EarlyStopping has patience 100 so it does not shorten the run. View `COMPARISON.md` for the manager-facing recommendation and evidence placeholders that must be completed after the required run.

## The Keras version

`keras_model.py`: architecture, PyTorch-to-Keras weight copy, validation equivalence check.

`train_keras.py`: `compile`, `fit`, callbacks, per-epoch history and best-epoch learning curve.

`compare.py`: 3-seed paired benchmarks written to `reports/comparison.csv`.

**Checkpoint policy:** `reports/*.keras` is deliberately ignored in Git to avoid committing large binary checkpoints. ModelCheckpoint writes them on rerun and saved-model reload is tested with a temporary `.keras` file.

## Tests

The 14 original Lab 07 tests remain unchanged; 12 new test cases cover output shape, parameter parity for multiple widths/layers, transposed weight copy, identical input rows, memorization, save/reload, restored-best validation loss, seed repeatability, and comparison CSV. Local smoke test: `26 passed` under Keras 3 using its **Torch backend**, **not** TensorFlow. The required official `ruff` and TensorFlow checks remain **unverified here** due to unavailable packages/network. Run them with Conda before submission.

## Limitations

Data are synthetic, not a real bike-share forecast. Both frameworks are subject to seed, initialization and hardware variability. The timed comparison includes fit only; model-build costs and installation overhead are not benchmarked. Keras and PyTorch may use different numerical kernels, shuffle ordering and initialization even with the same seed. Do not invent metrics or infer backend speed from provisional results.

## Submission

Repository **must** be named `lab08-same-model-twice`, with **at least 5 meaningful commits**. The first commit must contain the Lab 07 files *as originally supplied*, including `bikes_data.py` unchanged. Add Keras files in subsequent commits. `reports/` must be tracked; large checkpoints may remain ignored. Submit the **GitHub repository URL on Blackboard**; if private, share only with the instructor identified in the lab. See `SETUP_AND_SUBMISSION.md` for detailed Git commands and a final checklist.
