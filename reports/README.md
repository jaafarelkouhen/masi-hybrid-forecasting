# `reports/`

**Curated, human-facing deliverables** — as opposed to `outputs/`, which holds
raw machine artifacts.

```
reports/
├── executive_summary/   # high-level write-up of the results
├── figures/             # charts used in the report — gitignored, regenerate from the pipeline
└── final_results.md     # the final results table / headline metrics
```

`reports/` is what a reader (examiner, recruiter, stakeholder) looks at;
[`../outputs/`](../outputs/) is what the pipeline produces and the dashboard
consumes.
