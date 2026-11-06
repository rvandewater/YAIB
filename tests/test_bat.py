"""BAT must embed the measured variables (not their indicators) with their observation flag, and encode time as SeFT."""

import numpy as np
import torch

from icu_benchmarks.models.dl_models.bat import BAT, time_encoding

COLUMNS = ["age", "hr", "MissingIndicator_age", "MissingIndicator_hr", "counter"]


def make(**kwargs):
    torch.manual_seed(0)
    model = BAT((2, 6, len(COLUMNS)), num_classes=2, **kwargs)
    model.set_trained_columns(COLUMNS)
    return model


def test_indicators_are_observation_flags_not_variables():
    model = make(value_embed_size=4)
    x = torch.randn(2, 6, len(COLUMNS))
    x[..., 2:4] = 0.0
    model(x).sum().backward()
    used = model.sensor_embedding.weight.grad.abs().sum(dim=1) > 0
    assert used.tolist() == [True, True, False, False, True]
    # Marking the heart rate of the last hour as imputed changes that hour's prediction
    imputed = x.clone()
    imputed[:, -1, 3] = 1.0
    model.eval()
    assert not torch.allclose(model(imputed)[:, -1], model(x)[:, -1])


def test_odd_value_embedding_size_is_rounded_up():
    model = make(value_embed_size=3, depth=2, heads=2)
    assert model.value_embedding.out_features == 4 and model.merge.in_features % 4 == 0
    assert model(torch.randn(2, 6, len(COLUMNS))).shape == (2, 6, 2)


def test_time_encoding_matches_seft():
    steps, dim, max_time = 5, 6, 500.0
    timescales = max_time ** np.linspace(0, 1, dim // 2)
    scaled = np.arange(steps)[:, None] / timescales
    expected = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=-1)
    torch.testing.assert_close(time_encoding(steps, dim, max_time), torch.tensor(expected, dtype=torch.float))
