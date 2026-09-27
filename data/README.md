# data/

`raw/`, `interim/` and `processed/` are all git-ignored by default (see `.gitignore`) — nothing here is committed unless the dataset's licence explicitly permits redistribution **and** we've made an explicit decision to do so (Stage 2).

- `raw/` — files as downloaded, untouched.
- `interim/` — cleaned/resampled but not yet windowed or scaled (Stage 3 intermediate output).
- `processed/` — final windowed, scaled, split-ready arrays used to train and evaluate models.

Nothing lives here yet — Stage 2 (dataset acquisition and audit) hasn't started.
