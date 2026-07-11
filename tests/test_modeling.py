import numpy as np

from src.modeling import make_windows, monte_carlo_price_paths


def test_make_windows_shapes_and_values():
    series = np.arange(10, dtype=float)
    X, y = make_windows(series, window=3)
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    assert np.array_equal(X[0], [0.0, 1.0, 2.0])
    assert y[0] == 3.0
    assert np.array_equal(X[-1], [6.0, 7.0, 8.0])
    assert y[-1] == 9.0


def test_monte_carlo_price_paths_stay_positive_and_shaped():
    paths = monte_carlo_price_paths(
        last_price=100.0, daily_drift=0.0, daily_vol=0.05, horizon=20, n_sims=200, seed=1
    )
    assert paths.shape == (200, 20)
    assert np.all(paths > 0)
