"""Lab 08 tests: architecture, equivalence, preparation, Keras fit and comparison."""

import numpy as np
import pytest
from keras import models
from sklearn.metrics import mean_squared_error

from compare import COLUMNS, compare
from keras_model import build_keras_model, check_equivalence, copy_torch_weights
from model import BikeNet
from prepare import BikeDataset, prepare_data
from train import seed_everything
from train_keras import train_keras


def test_keras_one_output_per_row():
    model = build_keras_model(8, hidden=12)
    assert model.predict(np.zeros((7, 8), dtype=np.float32), verbose=0).shape == (7, 1)


@pytest.mark.parametrize("hidden,layers", [(8, 2), (16, 3), (64, 2)])
def test_keras_parameter_count_matches_torch(hidden, layers):
    a = build_keras_model(8, hidden=hidden, layers=layers)
    b = BikeNet(8, hidden=hidden, layers=layers)
    assert a.count_params() == sum(p.numel() for p in b.parameters())


def test_weight_copy_same_predictions(prepared):
    seed_everything(11)
    torch_model = BikeNet(prepared.x_train.shape[1], hidden=16)
    keras_model = build_keras_model(prepared.x_train.shape[1], hidden=16)
    copy_torch_weights(torch_model, keras_model)
    assert check_equivalence(torch_model, keras_model, prepared.x_valid) < 1e-5


def test_weight_copy_changes_dense_kernel(prepared):
    seed_everything(21)
    torch_model = BikeNet(prepared.x_train.shape[1], hidden=7)
    keras_model = build_keras_model(prepared.x_train.shape[1], hidden=7)
    copy_torch_weights(torch_model, keras_model)
    expected = torch_model.network[0].weight.detach().numpy().T
    np.testing.assert_allclose(keras_model.layers[0].get_weights()[0], expected)
    assert expected.shape != torch_model.network[0].weight.shape or not np.array_equal(
        expected, torch_model.network[0].weight.detach().numpy()
    )


def test_same_prepared_rows_for_both_frameworks(prepared):
    train = BikeDataset(prepared.x_train, prepared.y_train)
    np.testing.assert_array_equal(train.features.numpy(), prepared.x_train)
    np.testing.assert_array_equal(train.targets.numpy().reshape(-1), prepared.y_train)
    assert prepared.x_train.dtype == np.float32


def test_keras_can_memorize_one_small_batch():
    from keras import optimizers

    rng = np.random.default_rng(5)
    x = rng.normal(size=(12, 8)).astype(np.float32)
    y = (1.4 * x[:, 0] + 3.0).astype(np.float32)
    model = build_keras_model(8, hidden=32)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.02), loss="mse")
    first = float(model.evaluate(x, y, verbose=0))
    model.fit(x, y, batch_size=12, epochs=250, shuffle=False, verbose=0)
    last = float(model.evaluate(x, y, verbose=0))
    assert last < first * 0.05


def test_keras_save_reload_same_predictions(tmp_path):
    model = build_keras_model(8, hidden=8)
    x = np.random.default_rng(3).normal(size=(5, 8)).astype(np.float32)
    original = model.predict(x, verbose=0)
    filename = tmp_path / "model.keras"
    model.save(filename)
    loaded = models.load_model(filename)
    np.testing.assert_allclose(original, loaded.predict(x, verbose=0), atol=1e-6)


def test_best_weights_restored(tmp_path):
    data = prepare_data(seed=17, n=120)
    model, history, best_epoch, _ = train_keras(
        data,
        seed=17,
        epochs=5,
        hidden=8,
        batch_size=16,
        checkpoint=tmp_path / "best.keras",
    )
    assert len(history) == 5
    assert best_epoch == int(history.loc[history.val_loss.idxmin(), "epoch"])
    predictions = model.predict(data.x_valid, verbose=0).reshape(-1)
    assert mean_squared_error(data.y_valid, predictions) == pytest.approx(
        history.val_loss.min(), rel=1e-4, abs=1e-4
    )


def test_same_seed_repeats_short_run(tmp_path):
    data = prepare_data(seed=31, n=120)
    first, h1, _, _ = train_keras(
        data,
        seed=31,
        epochs=3,
        hidden=8,
        batch_size=16,
        checkpoint=tmp_path / "first.keras",
    )
    pred1 = first.predict(data.x_valid, verbose=0)
    second, h2, _, _ = train_keras(
        data,
        seed=31,
        epochs=3,
        hidden=8,
        batch_size=16,
        checkpoint=tmp_path / "second.keras",
    )
    np.testing.assert_allclose(
        pred1, second.predict(data.x_valid, verbose=0), atol=1e-4
    )
    np.testing.assert_allclose(h1.val_loss, h2.val_loss, atol=1e-4)


def test_comparison_csv_format(tmp_path):
    result = compare(seeds=(41,), epochs=2, n=120, output=tmp_path / "comparison.csv")
    assert list(result.columns) == COLUMNS
    assert set(result.framework) == {"Keras", "PyTorch"}
    assert len(result) == 2
    assert (result.seconds >= 0).all()
    assert (result.best_epoch >= 1).all()
    assert (tmp_path / "comparison.csv").is_file()
