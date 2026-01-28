# Notebooks Guide

Maintain one notebook per major task to keep outputs reviewable and lightweight:

1. `01_data_enrichment.ipynb` – schema review, exploratory checks, and documented additions to the unified dataset.
2. `02_eda.ipynb` – descriptive statistics, temporal coverage charts, inclusion trajectory plots, and infrastructure analyses.
3. `03_event_impact_modeling.ipynb` – event-indicator joins, association matrix construction, model experimentation.
4. `04_forecasting.ipynb` – access/usage forecasting, uncertainty quantification, and scenario analysis.

## Best Practices
- Start each notebook with a short markdown summary of purpose, data inputs, and outputs.
- Parameterize file paths through a shared config (e.g., `src/config.py`) to avoid hard-coded directories.
- Clear or strip excessive cell outputs before committing to keep diffs manageable.
- Mirror figures used in reports or dashboard under `reports/figures/` with descriptive filenames.
