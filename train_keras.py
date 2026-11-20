"""Keras compile/fit training on the EXACT Lab 07 prepared arrays."""

import time
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from keras_model import build_keras_model, check_equivalence, copy_torch_weights
from model import BikeNet
from prepare import SEED, prepare_data
from train import REPORTS, seed_everything

HIDDEN = 64
LAYERS = 2
EPOCHS = 100
LEARNING_RATE = 0.003
BATCH_SIZE = 128


def train_keras(
    data,
    seed=SEED,
    epochs=EPOCHS,
    hidden=HIDDEN,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    checkpoint=None,
):
    """Return best Keras model, per-epoch history, best epoch, elapsed seconds."""
    seed_everything(seed)
    keras.utils.set_random_seed(seed)
    model = build_keras_model(data.x_train.shape[1], hidden=hidden, layers=LAYERS)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    if checkpoint is None:
        checkpoint = REPORTS / "model_keras_best.keras"
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=epochs, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True
        ),
    ]
    start = time.perf_counter()
    history_object = model.fit(
        data.x_train,
        data.y_train.reshape(-1, 1),
        validation_data=(data.x_valid, data.y_valid.reshape(-1, 1)),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    history = pd.DataFrame(history_object.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history = history.rename(columns={"loss": "train_loss", "mae": "train_mae"})
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    return model, history, best_epoch, elapsed


def plot_keras_curves(history, best_epoch, output):
    """Commit this PNG alongside Lab 07 learning_curves.png."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.train_loss, label="Training MSE")
    ax.plot(history.epoch, history.val_loss, label="Validation MSE")
    ax.axvline(best_epoch, ls="--", color="black", label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE (rentals²)", title="Keras learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    data = prepare_data(seed=SEED)
    # Equivalence is checked with copied UNTRAINED weights, before any training.
    seed_everything(SEED)
    torch_net = BikeNet(data.x_train.shape[1], hidden=HIDDEN, layers=LAYERS)
    keras_net = build_keras_model(data.x_train.shape[1], hidden=HIDDEN, layers=LAYERS)
    copy_torch_weights(torch_net, keras_net)
    difference = check_equivalence(torch_net, keras_net, data.x_valid)
    print(f"Untrained weight-copy max difference: {difference:.9g}")

    model, history, best_epoch, seconds = train_keras(data, seed=SEED)
    history.to_csv(REPORTS / "history_keras.csv", index=False)
    plot_keras_curves(history, best_epoch, REPORTS / "learning_curves_keras.png")
    val_predictions = np.asarray(model.predict(data.x_valid, verbose=0)).reshape(-1)
    test_predictions = np.asarray(model.predict(data.x_test, verbose=0)).reshape(-1)
    print(
        f"Best epoch: {best_epoch}; validation MAE: "
        f"{mean_absolute_error(data.y_valid, val_predictions):.4f}; "
        f"test MAE: {mean_absolute_error(data.y_test, test_predictions):.4f}; "
        f"seconds: {seconds:.2f}"
    )
    print("Saved reports/history_keras.csv and reports/learning_curves_keras.png")


if __name__ == "__main__":
    main()
