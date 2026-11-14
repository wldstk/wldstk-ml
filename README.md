# wldstk-ml

Basic machine-learning project using only **numpy** and **pandas**.

## Layout

```
src/wldstk_ml/   package (data helpers, models, metrics)
tests/           unittest tests
data/raw/        input data (not versioned)
data/processed/  derived data (not versioned)
notebooks/       exploration
```

## Usage

```bash
make install       # editable install + black, isort, pyright
make test
make format        # isort + black
make lint          # isort/black check + pyright
make run           # demo on synthetic data
```
