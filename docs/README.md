# `docs/`

Written documentation for the engine. Start with
[`pipeline_index.md`](pipeline_index.md) for the current step-by-step status table.

| File | Content |
|---|---|
| [`index.md`](index.md) | Sphinx / Read the Docs entry point (master doc). |
| [`pipeline_index.md`](pipeline_index.md) | Pipeline status (steps 0→10), quick commands, final verdict. |
| [`methodology.md`](methodology.md) | Architecture, components, validation discipline. |
| [`anti_leakage.md`](anti_leakage.md) | The L1–L8 anti-leakage rules — the core methodological contract. |
| [`data_pipeline.md`](data_pipeline.md) | How data moves from raw to model-ready. |
| [`literature_review.md`](literature_review.md) | Background and references. |
| [`conf.py`](conf.py) + [`requirements.txt`](requirements.txt) | Sphinx config + build deps for Read the Docs. |

> `docs/references/` (PDFs of papers + cloned third-party repos) is a **private
> working bibliography** and is **gitignored** — it is not part of the published
> deliverable.

## Build the site

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

The hosted version on Read the Docs is driven by [`.readthedocs.yaml`](../.readthedocs.yaml)
at the repo root and rebuilds on every push to `main`.
