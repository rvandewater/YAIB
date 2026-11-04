"""GRU-D must follow the equations of Che et al. (2018) and train its decays, dropout included."""

import torch

from icu_benchmarks.models.dl_models.grud import GRUDLayer, decay_rate


def reference_grud(layer, x, observed):
    """Equations 2 and 10 to 16 of the paper, one sample and one hour at a time, with a unit time step between hours."""
    hidden = layer.hidden_dim
    W_z, W_r, W = layer.input_gates.weight.split(hidden)
    b_z, b_r, b = layer.input_gates.bias.split(hidden)
    V_z, V_r, V = layer.mask_gates.weight.split(hidden)
    U_z, U_r = layer.recurrent_gates.weight.split(hidden)
    U = layer.recurrent_candidate.weight
    outputs = torch.zeros(*x.shape[:2], hidden)
    for i in range(x.shape[0]):
        h = torch.zeros(hidden)
        for t in range(x.shape[1]):
            m = observed[i, t]
            # Last observation before t of each column (0, the mean, if none) and the hours since then
            delta, last = torch.full_like(m, float(t)), torch.zeros_like(m)
            for d in range(x.shape[2]):
                for s in reversed(range(t)):
                    if observed[i, s, d]:
                        delta[d], last[d] = t - s, x[i, s, d]
                        break
            gamma_x = torch.exp(-torch.relu(layer.input_decay_weight * delta + layer.input_decay_bias))
            gamma_h = torch.exp(-torch.relu(layer.hidden_decay.weight @ delta + layer.hidden_decay.bias))
            x_hat = m * x[i, t] + (1 - m) * gamma_x * last
            h_hat = gamma_h * h
            z = torch.sigmoid(W_z @ x_hat + U_z @ h_hat + V_z @ m + b_z)
            r = torch.sigmoid(W_r @ x_hat + U_r @ h_hat + V_r @ m + b_r)
            h_tilde = torch.tanh(W @ x_hat + U @ (r * h_hat) + V @ m + b)
            h = (1 - z) * h_hat + z * h_tilde
            outputs[i, t] = h
    return outputs


def test_matches_the_paper_equations():
    torch.manual_seed(0)
    layer = GRUDLayer(input_dim=3, hidden_dim=4).eval()
    with torch.no_grad():
        # Non-trivial decays and gate biases, so that every term matters
        for parameter in layer.parameters():
            parameter.uniform_(0.0, 0.5)
    x = torch.randn(2, 6, 3)
    observed = (torch.rand(2, 6, 3) < 0.5).float()
    observed[0, :, 0] = 0  # A column that is never observed
    torch.testing.assert_close(layer(x, observed), reference_grud(layer, x, observed))


def test_decays_train_from_the_start():
    # The decays start close to 1 (no decay) but must receive gradients: all-zero decay parameters would not, as relu has
    # no gradient at 0
    torch.manual_seed(0)
    layer = GRUDLayer(input_dim=3, hidden_dim=4)
    long_gap = torch.full((3,), 100.0)
    with torch.no_grad():
        assert decay_rate(layer.input_decay_weight * long_gap + layer.input_decay_bias).min() > 0.98
        assert decay_rate(layer.hidden_decay(long_gap)).min() > 0.98
    x = torch.randn(2, 6, 3)
    observed = (torch.rand(2, 6, 3) < 0.5).float()
    layer(x, observed).pow(2).sum().backward()
    for parameter in (layer.input_decay_weight, layer.input_decay_bias, *layer.hidden_decay.parameters()):
        assert parameter.grad.abs().sum() > 0


def test_dropout_is_applied_in_training_only():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 3)
    observed = (torch.rand(2, 6, 3) < 0.5).float()
    for dropout, recurrent_dropout in ((0.5, 0.0), (0.0, 0.5)):
        layer = GRUDLayer(input_dim=3, hidden_dim=4, dropout=dropout, recurrent_dropout=recurrent_dropout)
        assert not torch.allclose(layer.train()(x, observed), layer(x, observed))
        torch.testing.assert_close(layer.eval()(x, observed), layer(x, observed))
