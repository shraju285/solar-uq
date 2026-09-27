"""solaruq: uncertainty-aware deep learning for short-term solar power forecasting.

Sub-packages (filled in as each stage is implemented):
  data        - loading and cleaning raw sensor data (Stage 2-3)
  features    - feature engineering, scaling, windowing (Stage 3)
  models      - persistence, ARIMA, LSTM, quantile LSTM (Stage 4-6)
  uncertainty - CQR and other interval-calibration methods (Stage 7)
  evaluation  - shared metrics used by every model (Stage 4, 8)
  utils       - config loading, small shared helpers
"""

__version__ = "0.1.0"
