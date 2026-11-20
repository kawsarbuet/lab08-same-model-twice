"""Fair three-seed comparison: same Lab 07 prepared rows in both frameworks."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model import BikeNet
from prepare import make_loaders, prepare_data
from train import REPORTS, fit_network, predict_network, seed_everything
from train_keras import BATCH_SIZE, EPOCHS, HIDDEN, LEARNING_RATE, train_keras

SEEDS = (2026, 2027, 2028)
COLUMNS = ["framework", "seed", "best_epoch", "validation_mae", "test_mae", "seconds"]


def pytorch_run(data, seed, epochs=EPOCHS, hidden=HIDDEN):
    """Train original Lab 07 loop with precisely the main settings."""
    seed_everything(seed)
    train_loader, valid_loader, _ = make_loaders(
        data, batch_size=BATCH_SIZE, seed=seed
    )
    network = BikeNet(data.x_train.shape[1], hidden=hidden)
    start = time.perf_counter()
    _, best_epoch = fit_network(
        network,
        train_loader,
        valid_loader,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
    )
    elapsed = time.perf_counter() - start
    val_prediction = predict_network(network, data.x_valid)
    test_prediction = predict_network(network, data.x_test)
    return {
        "framework": "PyTorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": float(mean_absolute_error(data.y_valid, val_prediction)),
        "test_mae": float(mean_absolute_error(data.y_test, test_prediction)),
        "seconds": round(elapsed, 3),
    }


def keras_run(data, seed, epochs=EPOCHS, hidden=HIDDEN, checkpoint=None):
    """Train Keras on the same prepared rows and evaluate test once."""
    if checkpoint is None:
        checkpoint = REPORTS / f"model_keras_seed{seed}.keras"
    model, _, best_epoch, elapsed = train_keras(
        data, seed=seed, epochs=epochs, hidden=hidden, checkpoint=checkpoint
    )
    val_prediction = np.asarray(model.predict(data.x_valid, verbose=0)).reshape(-1)
    test_prediction = np.asarray(model.predict(data.x_test, verbose=0)).reshape(-1)
    return {
        "framework": "Keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": float(mean_absolute_error(data.y_valid, val_prediction)),
        "test_mae": float(mean_absolute_error(data.y_test, test_prediction)),
        "seconds": round(elapsed, 3),
    }


def compare(seeds=SEEDS, epochs=EPOCHS, n=6000, output=None):
    """One data preparation per seed, shared by both networks; six total runs."""
    rows = []
    for seed in seeds:
        data = prepare_data(seed=seed, n=n)
        # Identical data/splits/preprocessor for the paired models of this seed.
        rows.append(pytorch_run(data, seed=seed, epochs=epochs))
        rows.append(keras_run(data, seed=seed, epochs=epochs))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    return frame


def main():
    REPORTS.mkdir(exist_ok=True)
    comparison = compare(output=REPORTS / "comparison.csv")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Wrote reports/comparison.csv")


if __name__ == "__main__":
    main()
