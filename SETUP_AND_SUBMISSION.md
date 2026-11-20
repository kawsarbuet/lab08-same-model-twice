# AIGC5005 Lab 08: Setup and submission guide

## 1. GitHub first commit (important)

**Do not upload the whole ZIP as the final Blackboard submission.** Create a new repository `lab08-same-model-twice`. Instructor requires its **first commit to contain your Lab 07 files as they stood**. This ZIP contains both Lab 07 and Lab 08 files for convenience: to preserve the proper history you must stage **only the original Lab 07 files** in your first commit, then add Lab 08 files in later commits.

From the extracted `lab08-same-model-twice` folder:

```bash
git status
git clone https://github.com/kawsarbuet/lab08-same-model-twice
git branch -M main
cd lab08-same-model-twice

```

**For strict compliance**, the best method is: copy your original Lab 07 folder/repo into the new repo, commit it first without modifications (including its original README, requirements, and reports), then overlay the Lab 08 additions from this package. A history with an initial partial import may not meet the instructor's 'Lab 07 files as they stand' wording. Do not copy the embedded `.git` folder from the prior ZIP as part of the new repository.

```bash
git add .
git commit -m "Import original Lab 07 code and tests as baseline first commit "
git push
```
Next suggested commits (make at least five meaningful commits TOTAL):

```bash
git add keras_model.py tests/test_keras.py
 git commit -m "Add equivalent Keras architecture and parity tests"
git add train_keras.py
 git commit -m "Train Keras with callbacks and write learning curves"
git add compare.py
 git commit -m "Compare PyTorch and Keras over three seeds"
git add environment.yml requirements.txt requirements-dev.txt README.md COMPARISON.md SETUP_AND_SUBMISSION.md .gitignore reports/
 git commit -m "Document measured results and reproducibility checks"
```

Remove leading spaces before `git` when copying command lines if your terminal is sensitive. To connect the repository: `git remote add origin https://github.com/YOUR-USERNAME/lab08-same-model-twice.git`, then `git push -u origin main`.

## 2. Local Python / Conda steps

Open VS Code terminal in the repository folder. Install Miniconda if needed. Run:

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
python -c "import torch, tensorflow, keras"
python train.py
python train_keras.py
python compare.py
python -m ruff format .
python -m ruff format --check .
python -m ruff check .
python -m ruff check . --fix
python -m pytest
```

If the Ruff lint command reports an issue, use the reported location to correct it and rerun all checks. **Only after all commands pass** commit regenerated `reports/history_keras.csv`, `reports/learning_curves_keras.png`, and `reports/comparison.csv` (plus any other regenerated Lab07 reports). A CUDA/GPU build is not required; a CPU TensorFlow install is fine when properly supported for your OS.

## 3. Finish COMPARISON.md

Read `reports/comparison.csv`, calculate the PyTorch/Keras mean test MAE and mean seconds. Replace `[FILL]` markers in `COMPARISON.md` and `[PENDING]` labels in README. The final 250–400-word document must include at least three evidenced differences, a scenario-specific recommendation, a trade-off, and a condition that changes the recommendation. Do **not** use preview Keras Torch-backend results as official TensorFlow measurements.

## 4. Fresh-clone final check

Clone the GitHub repository into a new directory, create its Conda environment, and rerun `ruff format --check`, `ruff check`, `pytest`, `train.py`, `train_keras.py`, and `compare.py`. Check that `reports/comparison.csv` contains six rows and both learning curve PNGs are tracked by Git. Check `git status`, `git log --oneline`, and that the GitHub URL is accessible. Submit the GitHub URL on Blackboard before the posted deadline.

**No deployment is required.**
