# Lakeshore Bike Demand: The Same Model in PyTorch and Keras

**AIGC 5005 — Lab 08: The Same Model, Twice**  
**Repository:** [lab08-same-model-twice](https://github.com/kawsarbuet/lab08-same-model-twice)  
**Scope:** Reproduce and compare the original Lab 07 PyTorch bike-demand regressor with an equivalent Keras 3 model. No deployment is required.

## What it does

Lakeshore Bike Share needs to choose a maintainable framework for a potential weekly training job. This repository preserves the Lab 07 PyTorch data generation, models, training loop, experiments, baselines, and reports; adds an equivalent configurable Keras network; checks functional equivalence by copying weights before training; and compares both frameworks across three seeds.

The main entry points are `train.py` (PyTorch), `train_keras.py` (Keras), and `compare.py` (paired comparison). The decision and trade-offs are explained in [COMPARISON.md](COMPARISON.md).

## How to run it

From the repository root, install and activate the specified Python 3.12 Conda environment:

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
```

If the environment already exists, activate it rather than creating it again. The environment uses `conda-forge` and the pip requirements; see `requirements.txt` and `requirements-dev.txt` for the exact project dependency pins.

**Verify the Keras backend before running training or tests.** The assignment requires Keras 3 with **TensorFlow**, not Keras 3 running on the PyTorch backend:

```bash
python -c "import torch, tensorflow as tf, keras; print('PyTorch:', torch.__version__); print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__); print('Keras backend:', keras.backend.backend())"
```

The backend line must say `Keras backend: tensorflow`. If it does not, configure `KERAS_BACKEND=tensorflow` **before starting a new Python process**, then check again. An `import tensorflow` success alone does not prove Keras uses that backend.

Run the three project scripts:

```bash
python train.py
python train_keras.py
python compare.py
```

- **`train.py`** reruns the original Lab 07 PyTorch experiments, plots, baselines, and `reports/metrics.json`.
- **`train_keras.py`** performs the weight-copy equivalence check, trains the main Keras model, and writes `reports/history_keras.csv` and `reports/learning_curves_keras.png`.
- **`compare.py`** prepares paired data and trains the main PyTorch and Keras configurations on seeds **2026, 2027, and 2028**, producing six rows in `reports/comparison.csv` and per-seed Keras checkpoints.

The full comparison trains six models and may take several minutes depending on hardware and backend.

## Data and split

`bikes_data.py` produces **6,000 synthetic station-hour observations**. `prepare.py` creates disjoint **60% train (3,600), 20% validation (1,200), and 20% test (1,200)** splits for a chosen seed.

The numerical variables `hour`, `temperature_c`, `rain_mm`, `wind_kph`, and `is_weekend` are standardized; `station_type` is one-hot encoded. This produces **eight `float32` inputs**. Preprocessing is fitted on training rows only and then applied to the held-out splits.

For each seed, `compare.py` prepares the arrays **once** and shares the same rows and preprocessing between the two frameworks. Different seeds generate different splits; within each paired run the splits are the same.

## Model and training

Both main implementations use the following architecture and training settings:

| Setting | PyTorch and Keras main run |
|---|---|
| Input | 8 prepared features |
| Hidden layers | 64 ReLU units, then 64 ReLU units |
| Output | One linear regression value, no activation |
| Parameters | 4,801 |
| Loss | Mean squared error (MSE) |
| Optimizer | Adam |
| Learning rate | 0.003 |
| Batch size | 128 |
| Training duration | 100 epochs |
| Model selection | Lowest validation loss |

The constructors support configurable hidden widths/layers. PyTorch uses an explicit loop (`zero_grad`, `backward`, `step`) and restores its best-validation `state_dict`. Keras uses `compile()` and `fit()`, `EarlyStopping(restore_best_weights=True, patience=epochs)`, and `ModelCheckpoint(monitor="val_loss", save_best_only=True)`. The long patience avoids early termination in the main comparison.

**Equivalence before training:** `copy_torch_weights()` copies PyTorch `Linear` weights into Keras `Dense` layers, **transposing** `(outputs, inputs)` to `(inputs, outputs)`. `check_equivalence()` evaluates both on the same validation rows with tolerance `1e-5`. This checks whether they implement the same function after copying weights; the independent benchmark runs do **not** start from identical framework weights.

Each run evaluates its selected final model on the test rows once. Keras predictions shaped `(N, 1)` are flattened to `(N,)` before MAE calculation to avoid unintended broadcasting.

## Results

### Retained Lab 07 results

The recorded Lab 07 `reports/metrics.json` selected a **wide-network** experiment, with validation MAE **2.4554** and test MAE **2.4228** rentals. This is **not** the Lab 08 64-unit main-network comparison; its numbers must not be compared as if the model configuration were identical.

### Lab 08: Recorded results over three seeds

The previously supplied `reports/comparison.csv` contains the following six recorded runs:

| Framework | Seed | Best epoch | Validation MAE | Test MAE | Training seconds |
|---|---:|---:|---:|---:|---:|
| PyTorch | 2026 | 96 | 2.7400 | 2.6815 | 28.423 |
| Keras | 2026 | 95 | 2.7180 | 2.5738 | 40.630 |
| PyTorch | 2027 | 99 | 2.5233 | 2.6808 | 23.432 |
| Keras | 2027 | 99 | 2.6171 | 2.6859 | 64.288 |
| PyTorch | 2028 | 100 | 2.7478 | 2.8100 | 32.708 |
| Keras | 2028 | 98 | 2.6649 | 2.6877 | 62.730 |

| Framework | Mean test MAE | Mean training seconds |
|---|---:|---:|
| PyTorch | **2.7241** | **28.188** |
| Keras | **2.6491** | **55.883** |

In these recorded runs, Keras had **0.0750 rentals lower mean test MAE**, while PyTorch took approximately **27.695 fewer seconds** per training run on average. These are descriptive results for three seeds, not a universal accuracy or speed ranking. Fit-time measurements exclude setup and the broader weekly workflow.

The previously supplied `reports/history_keras.csv` records 100 epochs and its lowest-validation-loss epoch is **95**, consistent with the Keras/2026 comparison row. The Keras learning-curve PNG was present in the supplied project materials.

**Result provenance caveat:** The archived CSV and history do not record the active Keras backend. Earlier local evidence showed a successful TensorFlow import, but that alone does **not** establish that these *specific* saved runs used the TensorFlow backend. Before final submission, verify the active backend and regenerate `train_keras.py` and `compare.py` reports if necessary; update this table if rerun values change.

See [COMPARISON.md](COMPARISON.md) for the manager-facing framework recommendation, evidence, trade-offs, and conditions for reconsidering it.

## Experiments

The retained Lab 07 experiments compare the original main model, a lower learning rate, a wider network, and mean/linear-regression baselines. Their historical metrics, curves, and plots are kept under `reports/`.

Lab 08 compares **only the main 64-by-64 network**, using matching preprocessing and paired splits, MSE, Adam, learning rate, batch size, and epoch budget. Framework-specific initialization, minibatch ordering, and numerical kernels can still cause differences.

## The Keras version

- **`keras_model.py`** builds the configurable equivalent network, transfers PyTorch weights, and checks prediction equivalence.
- **`train_keras.py`** implements `compile()`/`fit()`, validation-based callbacks, history logging, and learning-curve plotting.
- **`compare.py`** runs the three paired-seed experiments and writes the CSV used in the comparison.

**Checkpoint policy:** `.gitignore` excludes `reports/*.keras` and `reports/*.h5` binary checkpoints, **not the `reports/` directory**. Training recreates those checkpoints. CSV, PNG, and other required report artifacts should be tracked and committed.

## Tests

The supplied project materials describe **14 retained Lab 07 tests and 12 additional Lab 08 test cases**, including parametrized architecture checks. They cover output shape, parameter-count parity, the transposed weight copy and matching predictions, identical prepared rows, small-batch memorization, save/reload, restoration of best weights, repeatability for a fixed seed, and comparison CSV structure.

A previous terminal screenshot reported **26 passed, 111 warnings**. This is a historical observation, **not** proof of a final fresh-clone pass under the required TensorFlow backend. Passing current Ruff checks has also **not** been independently confirmed here.

After activating the Python 3.12 environment, verify `keras.backend.backend() == "tensorflow"`, then run all required checks:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

All commands must exit successfully. Review warnings and fix relevant issues before submission. Re-run the three training scripts and verify the recreated `reports/` files agree with this README and `COMPARISON.md`.

## Limitations

- The dataset is synthetic, not a validation of real operational bike-share demand.
- Three seeds give limited evidence; performance may vary with data, hardware, and software versions.
- Different defaults for initialization, data order, and numerical operations limit exact training parity, even though copied-weight predictions are equivalent.
- Reported training seconds cover the measured training phase, not setup, startup, or full weekly-job cost.
- The archived metrics lack explicit TensorFlow-backend provenance; confirm and rerun the comparison before relying on them as TensorFlow-backed results.

## Submission

The assignment calls for one GitHub repository named `lab08-same-model-twice`, with at least five meaningful commits and the original Lab 07 project in the first commit. Preserve `bikes_data.py` unchanged. Commit the required Keras history and plots, comparison CSV, source, tests, environment files, and this README; do not ignore the entire `reports/` folder.

The previously supplied Git archive showed an initial commit before the Lab 07 import commit, so verify the **actual remote history** and resolve any instructor concern about first-commit ordering without casually rewriting shared history. Also review `git status` for report files that have not been committed.

Before submitting, **fresh-clone** the repository and verify the Conda environment, TensorFlow backend, all three checks, and all three scripts. Submit the repository URL in Blackboard. If private, grant instructor access according to the assignment instructions.
