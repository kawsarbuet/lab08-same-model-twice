# Lakeshore Bike Demand: the same model in PyTorch and Keras

**AIGC 5005 — Lab 08: The Same Model, Twice**  
**Repository:** `lab08-same-model-twice`  
**Purpose:** Compare the original Lab 07 PyTorch network with an equivalent Keras 3 implementation, using the same prepared rows and training settings. This project does **not** deploy a model.

## What it does

The project preserves Lab 07's synthetic bike-demand regression workflow and implements the same neural-network architecture using Keras. It checks **functional equivalence before training** by transferring PyTorch weights to Keras and comparing validation predictions. It then trains the two frameworks independently on three reproducible data seeds and records validation error, held-out test error and training time.

The original `bikes_data.py`, PyTorch training loop, Lab 07 experiments, reports and tests remain in the project. The Keras implementation is in `keras_model.py` and `train_keras.py`; `compare.py` produces the framework comparison.

## How to run it

Open a terminal in the repository root, using Conda with Python 3.12:

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
```

If the environment already exists, activate it instead of creating it again. The Conda file uses `conda-forge` and pip requirements. The packaged requirements pin **PyTorch 2.10.0, TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, matplotlib 3.10.8, pytest 9.0.2 and Ruff 0.14.8**. These are the versions specified by the repository, not independently verified versions on every machine.

**Confirm the Keras backend before running any training:**

```bash
python -c "import torch, tensorflow as tf, keras; print('PyTorch:', torch.__version__); print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__); print('Keras backend:', keras.backend.backend())"
```

The final line must report **`Keras backend: tensorflow`**. Successful `import tensorflow` alone does not prove Keras is using TensorFlow. If another backend is selected, configure `KERAS_BACKEND=tensorflow` **before Python starts**, then repeat the check.

Run the three entry points from the repository root:

```bash
python train.py
python train_keras.py
python compare.py
```

- `train.py` recreates the original Lab 07 PyTorch training experiments, plots, baselines and `reports/metrics.json`.
- `train_keras.py` checks weight-copy equivalence and trains the main Keras model, recreating `reports/history_keras.csv` and `reports/learning_curves_keras.png`.
- `compare.py` trains both main configurations on seeds **2026, 2027 and 2028**, then writes `reports/comparison.csv` (six rows). It also writes per-seed Keras checkpoints.

Recreate outputs on a **fresh clone** before submission. The six-run comparison can take several minutes, depending on the computer and backend.

## Data and split

`bikes_data.py` generates **6,000 synthetic station-hour observations**. `prepare.py` uses the selected seed to create disjoint **60% training (3,600 rows), 20% validation (1,200 rows), and 20% test (1,200 rows)** subsets.

The five numerical input variables (`hour`, `temperature_c`, `rain_mm`, `wind_kph`, `is_weekend`) are standardized, while `station_type` is one-hot encoded. The resulting **eight float32 features** are shared between PyTorch and Keras. The `StandardScaler` and `OneHotEncoder` are **fitted on training rows only**, then applied to validation and test rows. There is no independent Keras preprocessing path.

For each comparison seed, `compare.py` prepares the data **once**, passing the same arrays to both models. Splits differ across seeds, but are identical **within each paired PyTorch/Keras run**.

## Model and training

Both implementations use:

- Input: **8** prepared features.
- Hidden layer 1: **64 units, ReLU**.
- Hidden layer 2: **64 units, ReLU**.
- Output: **1 linear value**, no output activation.
- Total model parameters: **4,801** for the main configuration.
- Loss: **mean squared error (MSE)**; optimizer: **Adam**; learning rate: **0.003**.
- Batch size: **128**; epochs: **100**; model selection: **minimum validation loss**.

Hidden width and number of layers can be varied in the model constructors. PyTorch implements the training loop explicitly with `zero_grad`, `backward` and `step`; Keras uses `compile()` and `fit()`. PyTorch manually restores the best-validation-loss `state_dict`. Keras uses `EarlyStopping(patience=epochs, restore_best_weights=True)` and `ModelCheckpoint(monitor="val_loss", save_best_only=True)`. Patience equal to the epoch count prevents early termination in the main comparison.

**Equivalence check:** `copy_torch_weights()` transfers *untrained* PyTorch `Linear` weights to Keras `Dense` layers, transposing each weight matrix from `(outputs, inputs)` to `(inputs, outputs)`. `check_equivalence()` compares predictions on the **same validation rows** with a tolerance of **1e-5**. This establishes architectural/function equivalence; the fair training runs use separately initialized models rather than identical copied starting weights.

Both versions score their final best-validation model on test rows once per run. Keras `(N, 1)` predictions are flattened to `(N,)` before calculating MAE.

## Results

### Lab 07 retained results (not the Lab 08 main-model benchmark)

The packaged `reports/metrics.json` identifies the **wide-network** experiment as Lab 07's validation-selected model and reports **test MAE 2.4228 rentals** (validation MAE 2.4554). This model used a different hidden width and **must not** be compared as though it were the Lab 08 main 64-unit network.

### Lab 08 — recorded three-seed results

The uploaded repository includes `reports/comparison.csv` with these six results:

| Framework | Seed | Best epoch | Validation MAE | Test MAE | Training seconds |
|---|---:|---:|---:|---:|---:|
| PyTorch | 2026 | 96 | 2.7400 | 2.6815 | 28.423 |
| Keras | 2026 | 95 | 2.7180 | 2.5738 | 40.630 |
| PyTorch | 2027 | 99 | 2.5233 | 2.6808 | 23.432 |
| Keras | 2027 | 99 | 2.6171 | 2.6859 | 64.288 |
| PyTorch | 2028 | 100 | 2.7478 | 2.8100 | 32.708 |
| Keras | 2028 | 98 | 2.6649 | 2.6877 | 62.730 |

| Framework | Mean test MAE across three seeds | Mean training seconds |
|---|---:|---:|
| **PyTorch** | **2.7241** | **28.188** |
| **Keras** | **2.6491** | **55.883** |

**Interpretation:** In these recorded runs, Keras has a lower mean test MAE by approximately **0.0750 rentals**; PyTorch has a lower mean fit duration by approximately **27.695 seconds**. These are descriptive observations on three seeds, not proof that one framework is generally more accurate or faster. Timing depends on hardware and backend, and the comparison times only the fit stage.

The packaged `reports/history_keras.csv` contains **100 epochs** and identifies **epoch 95** as the lowest-validation-loss epoch, consistent with the Keras/2026 row in `comparison.csv`. The Keras learning-curve PNG is included. **The archived report files do not themselves record the active Keras backend.** A local environment has been shown to import TensorFlow, but confirmation that *these exact artifacts* were produced with the TensorFlow backend requires checking the backend and, if necessary, rerunning `train_keras.py` and `compare.py` with `Keras backend: tensorflow` displayed. Replace the report files and record the checked backend if you rerun them.

See `COMPARISON.md` for the manager-focused framework recommendation. **Its original placeholder metrics/status must also be updated to agree with the measured CSV before submission.**

## Experiments

The preserved Lab 07 PyTorch experiments compare the `main` network, a low-learning-rate run, a wider network, and mean/linear-regression baselines. Their histories, figures and metrics remain under `reports/`.

Lab 08 compares only the **main two-hidden-layer, 64-unit architecture** across seeds 2026, 2027 and 2028. It uses the same data split and preprocessing per paired seed, loss, optimizer family, learning rate, batch size and 100-epoch duration. Different initialization strategies, mini-batch orders and backend numerical kernels remain possible sources of differences.

## The Keras version

- `keras_model.py` builds the architecture and contains weight transfer and prediction equivalence checks.
- `train_keras.py` performs `compile`, `fit`, callback-based best-model selection, per-epoch history recording, and learning-curve plotting.
- `compare.py` orchestrates paired framework runs and exports `reports/comparison.csv`.

**Checkpoint policy:** `.gitignore` intentionally excludes `reports/*.keras` and `reports/*.h5` binary model checkpoints; **it does not ignore `reports/`**. Re-running training recreates checkpoint files. Required report CSV/PNG files should be committed. The uploaded ZIP contains local `.keras` files, but the Git-tracked repository list does not include them.

## Tests

The project includes the **14 original Lab 07 tests** and **12 additional Lab 08 test cases** (including three parametrized parameter-count cases), covering:

- One prediction per input row and parameter parity across several architectures.
- PyTorch-to-Keras transposed weight copy and prediction equivalence.
- Shared prepared rows and float32 inputs.
- Small-batch memorization, Keras save/reload, and best-weight restoration.
- Repeating a short training run with the same seed.
- Comparison CSV format and required fields.

**Observed local check:** The user-provided terminal screenshot shows **26 passed, 111 warnings**. TensorFlow import also succeeded in the activated Conda environment. These observations do not by themselves verify that Keras used TensorFlow for every test or for the saved comparison runs.

Run all three mandatory checks in the **TensorFlow-backed Conda environment**:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

All must exit successfully. A previous `I001` import-order issue was reported and a `Fix Keras imports and Ruff linting` commit exists in the supplied Git history, but **successful current Ruff outputs were not provided or verified in this review**. If pytest emits warnings, review them even if all tests pass. Verify the backend in the same terminal before running tests.

## Limitations

- The dataset is simulated; findings are not validated on real bike-share demand data.
- Three seeds are useful for comparison but are not enough to generalize broadly.
- Different framework initializers, shuffling behavior, numerical kernels and hardware can affect results.
- Timings measure `fit`/training time, not installation, startup, model construction or end-to-end scheduling.
- The supplied CSV/history do not establish their backend provenance. Verify TensorFlow and regenerate reports when necessary.
- `COMPARISON.md` still requires updating from its original placeholders before final submission.

## Submission

The instructor requires a GitHub repository named **`lab08-same-model-twice`**, at least **five meaningful commits**, and a first commit containing the original Lab 07 files. Keep `bikes_data.py` unchanged. Commit the Keras plot, histories, comparison CSV and other required report files; `.gitignore` must not exclude the whole `reports/` folder.

**Review the Git history:** The supplied archive contains a Git commit titled `Initial commit` *before* `Import original Lab 07 code and tests as baseline first commit`. Although the archive has more than five commits, its first-commit sequence appears not to match the instructor's literal instruction. Check the actual remote history and your instructor's expectations before submission; do not rewrite a shared repository's history without understanding the consequences.

**Also commit current report changes:** The archive's Git working tree shows modified report CSV/JSON files that are not reflected in the recorded commit. Review `git status`, then commit the final validated report outputs as appropriate.

Run all scripts and checks from a fresh clone, confirm that the required artifacts are present in GitHub, and submit the **repository URL** on Blackboard. If the repository is private, grant access only to the instructor specified in the assignment. See `SETUP_AND_SUBMISSION.md` for the procedural checklist.
