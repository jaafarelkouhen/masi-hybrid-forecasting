# `tests/`

Test suite for the `masi_hybrid_forecasting` package.

```
tests/
├── unit/          # pure unit tests, no disk/data I/O
└── integration/   # tests that touch the filesystem / real data artifacts
```

Run everything:

```bash
pip install -e ".[dev]"
python -m pytest
```

Run only the fast units:

```bash
python -m pytest tests/unit
```

Integration tests expect the pipeline artifacts to exist under
[`../outputs/`](../outputs/); run the relevant `scripts/` step first if a test
reports a missing input.
