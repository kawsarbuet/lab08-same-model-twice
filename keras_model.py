"""Keras 3 equivalent of Lab 07 BikeNet; weight transfer for proof."""

import keras
import numpy as np
import torch
from torch import nn


def build_keras_model(in_features: int, hidden: int = 64, layers: int = 2):
    """Exactly [Dense(hidden), ReLU] * layers then Dense(1)."""
    if layers < 2:
        raise ValueError("At least two hidden layers are required")
    model = keras.Sequential(name="bike_demand_keras")
    model.add(keras.Input(shape=(in_features,)))
    for _ in range(layers):
        model.add(keras.layers.Dense(hidden, activation="relu"))
    model.add(keras.layers.Dense(1))
    return model


def copy_torch_weights(torch_model: nn.Module, keras_model) -> None:
    """Copy every Linear to Dense; PyTorch W=(out,in), Keras W=(in,out)."""
    torch_layers = [m for m in torch_model.modules() if isinstance(m, nn.Linear)]
    keras_layers = [m for m in keras_model.layers if isinstance(m, keras.layers.Dense)]
    if len(torch_layers) != len(keras_layers):
        raise ValueError("Model layer counts do not match")
    for torch_layer, keras_layer in zip(torch_layers, keras_layers, strict=True):
        weight = torch_layer.weight.detach().cpu().numpy().T
        bias = torch_layer.bias.detach().cpu().numpy()
        if tuple(keras_layer.kernel.shape) != weight.shape:
            raise ValueError("Dense kernel shape differs from transposed weight")
        keras_layer.set_weights([weight, bias])


def check_equivalence(torch_model, keras_model, valid_features, tolerance=1e-5):
    """Report worst absolute error on the SAME validation rows."""
    torch_model.eval()
    with torch.no_grad():
        torch_predictions = (
            torch_model(torch.as_tensor(valid_features, dtype=torch.float32))
            .cpu()
            .numpy()
            .reshape(-1)
        )
    keras_predictions = np.asarray(
        keras_model.predict(valid_features, verbose=0)
    ).reshape(-1)
    maximum_difference = float(np.max(np.abs(torch_predictions - keras_predictions)))
    if maximum_difference > tolerance:
        raise AssertionError(
            f"Models differ by {maximum_difference:.8g}; tolerance is {tolerance}"
        )
    return maximum_difference
