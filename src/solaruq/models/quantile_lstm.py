"""Direct multi-step, multi-quantile LSTM forecaster (Stage 6).

Same encoder as Stage 5's `LSTMPointForecaster` (src/solaruq/models/lstm.py):
one LSTM layer over the 6-hour lookback, its final hidden state concatenated
with the flattened horizon-known solar position, one linear layer mapping
straight to every horizon step at once. The only structural change is the
output width -- instead of 8 values (one per horizon step) the head
produces 8 x n_quantiles values, reshaped to (batch, horizon_steps,
n_quantiles) -- and a monotonic transform on top of it.

Non-crossing by construction: trained independently, per-quantile pinball
losses give no guarantee that predicted quantiles come out in the right
order (e.g. the P90 forecast could land below the P50 forecast for some
window). Rather than leaving that possibility in and reporting how often it
happens, the head's raw output is turned into non-decreasing quantiles by
construction: the lowest quantile is used as-is, and each subsequent
quantile is the previous one plus a non-negative increment
(`softplus(raw)`, so it's always >= 0 but never clipped to exactly 0 the
way `relu` would be). This only works because `quantiles` is sorted
ascending, which both the config and `QuantileLSTMForecaster.__init__`
assume and check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLSTMForecaster(nn.Module):
    def __init__(
        self,
        n_lookback_features: int,
        n_horizon_known_features: int,
        horizon_steps: int,
        quantiles: list[float],
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantiles must be sorted ascending for the non-crossing head to be valid")
        self.horizon_steps = horizon_steps
        self.quantiles = list(quantiles)
        n_quantiles = len(self.quantiles)
        self.lstm = nn.LSTM(input_size=n_lookback_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + n_horizon_known_features * horizon_steps, horizon_steps * n_quantiles)

    def forward(self, X_lookback: torch.Tensor, X_horizon_known: torch.Tensor) -> torch.Tensor:
        """X_lookback: (batch, lookback_steps, n_lookback_features).
        X_horizon_known: (batch, horizon_steps, n_horizon_known_features).
        Returns: (batch, horizon_steps, n_quantiles), non-decreasing along
        the last axis (in the same order as `self.quantiles`).
        """
        _, (h_n, _) = self.lstm(X_lookback)
        h_last = self.dropout(h_n[-1])
        horizon_flat = X_horizon_known.reshape(X_horizon_known.shape[0], -1)
        combined = torch.cat([h_last, horizon_flat], dim=1)
        raw = self.head(combined).view(-1, self.horizon_steps, len(self.quantiles))

        lowest = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        return torch.cat([lowest, increments], dim=-1).cumsum(dim=-1)


def pinball_loss(y_true: torch.Tensor, y_pred_quantiles: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """y_true: (batch, horizon_steps). y_pred_quantiles: (batch, horizon_steps, n_quantiles).
    Mean pinball (quantile) loss over every (window, horizon step, quantile) triple.
    """
    q = torch.as_tensor(quantiles, dtype=y_pred_quantiles.dtype, device=y_pred_quantiles.device).view(1, 1, -1)
    error = y_true.unsqueeze(-1) - y_pred_quantiles
    return torch.maximum(q * error, (q - 1) * error).mean()
