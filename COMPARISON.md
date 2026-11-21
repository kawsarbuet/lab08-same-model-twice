# PyTorch vs Keras — Lakeshore Bike Demand

**AIGC 5005 · Lab 08: The Same Model, Twice**

## Evidence and observations

Both implementations use the same prepared bike-demand rows, network structure, loss, optimizer settings, batch size, and epoch budget. In `keras_model.py`, the PyTorch weights are copied into Keras with each dense kernel transposed. The equivalence check compares predictions on shared validation rows within **1e-5**. This demonstrates that the models can compute the same function **before training**; separately seeded training does not give them identical starting weights.

**First, training code and control differ.** `train.py` explicitly manages PyTorch's gradient steps (`zero_grad`, `backward`, `step`) and restores the best validation-loss weights. `train_keras.py` uses `compile()` and `fit()`, with `EarlyStopping(restore_best_weights=True)` and `ModelCheckpoint(save_best_only=True)`. Keras reduces loop-management code; PyTorch exposes individual operations for debugging and customization.

**Second, recorded accuracy differs modestly.** In `reports/comparison.csv`, the mean held-out test MAE over seeds **2026, 2027, and 2028** is **2.7241** for PyTorch and **2.6491** for Keras, a **0.0750-rental** advantage for Keras. Keras records a lower test MAE in two of three runs. Three synthetic-data seeds do not establish a general accuracy advantage.

**Third, recorded training time differs.** Mean training durations are **28.188 seconds** for PyTorch and **55.883 seconds** for Keras. These measure training on the recorded setup, not deployment or total workflow time. Framework backend, hardware, and numerical kernels can affect timings.

## Recommendation to the team

I recommend **Keras for Lakeshore Bike Share's initial weekly retraining job**. Its built-in callbacks, checkpointing, and concise `fit()` workflow reduce maintenance burden for a two-person team, while the recorded error is comparable to—or slightly below—PyTorch. The team gives up the fine-grained training-loop control and lower measured training time demonstrated by PyTorch.

**I would change this recommendation to PyTorch** if the planned Hugging Face pretrained model requires a PyTorch-first integration, or if repeated production-like benchmarks show that weekly runtime is a material constraint. Before finalizing the choice, rerun `compare.py` with the **TensorFlow Keras backend explicitly verified**: the saved comparison CSV does not independently prove which Keras backend produced it. Neither framework is deployed in this lab.

