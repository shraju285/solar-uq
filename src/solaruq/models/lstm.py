"""Direct multi-step LSTM point-forecast model.

Deliberately the simplest architecture that can use everything Stage 3
already built: one LSTM layer reads the 6-hour lookback (target history +
observed WS_1 weather + solar position), its final hidden state is
concatenated with the horizon's own solar-position values (the only
"knowable in advance" information about the future 2 hours), and a single
linear layer maps that straight to all 8 horizon power values at once.

"Direct" multi-step means exactly that one-shot mapping -- no recursive
step-by-step forecasting (where step 2's forecast would need step 1's
*predicted* value as input, compounding error through the horizon). No
GRU, attention, or second LSTM over the horizon: the horizon's solar
position is flattened and concatenated, not sequence-modelled, keeping
this a single small recurrent layer plus a linear head.
"""

import torch
import torch.nn as nn


class LSTMPointForecaster(nn.Module):
    def __init__(
        self,
        n_lookback_features: int,
        n_horizon_known_features: int,
        horizon_steps: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.lstm = nn.LSTM(input_size=n_lookback_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + n_horizon_known_features * horizon_steps, horizon_steps)

    def forward(self, X_lookback: torch.Tensor, X_horizon_known: torch.Tensor) -> torch.Tensor:
        """X_lookback: (batch, lookback_steps, n_lookback_features).
        X_horizon_known: (batch, horizon_steps, n_horizon_known_features).
        Returns: (batch, horizon_steps) -- all horizon steps in one pass.
        """
        _, (h_n, _) = self.lstm(X_lookback)
        h_last = self.dropout(h_n[-1])  # (batch, hidden_size), final layer's final hidden state
        horizon_flat = X_horizon_known.reshape(X_horizon_known.shape[0], -1)
        combined = torch.cat([h_last, horizon_flat], dim=1)
        return self.head(combined)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
