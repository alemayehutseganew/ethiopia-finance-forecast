# Ethiopia Financial Inclusion Forecasting

This repository hosts the Week 10 challenge for the 10 Academy Artificial Intelligence Mastery program. The goal is to design an end-to-end forecasting system that explains and projects Ethiopia's financial inclusion trajectory across two World Bank Global Findex pillars:

1. **Access (Account Ownership Rate)** – Share of adults (15+) with a financial institution or mobile-money account.
2. **Usage (Digital Payment Adoption)** – Share of adults who made or received a digital payment in the past 12 months.

## Project Goals
- Consolidate and enrich the unified financial inclusion dataset for Ethiopia.
- Perform exploratory analysis to surface inclusion drivers, gaps, and risks.
- Quantify the impacts of policies, market entries, and infrastructure shocks on indicators.
- Produce forecasts for Access and Usage covering 2025-2027 under multiple scenarios.
- Deliver an interactive Streamlit dashboard for stakeholders to explore data, events, and projections.

## Implementation Overview
- **Task 1 – Data Exploration & Enrichment:** Curated additional macro-financial indicators and event signals logged in [data_enrichment_log.md](data_enrichment_log.md); merged them with the base dataset through staging pipelines inside [notebooks/01_data_enrichment.ipynb](notebooks/01_data_enrichment.ipynb).
- **Task 2 – Exploratory Data Analysis:** Profiled temporal trends, regional disparities, and correlation structures using reusable plotting helpers from [src/event_effects.py](src/event_effects.py); key visuals live in [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb) and exports under [reports/figures](reports/figures).
- **Task 3 – Event Impact Modeling:** Quantified structural breaks and lagged elasticities with intervention regression blocks in [notebooks/03_event_impact_modeling.ipynb](notebooks/03_event_impact_modeling.ipynb), supported by helper routines in [src/__init__.py](src/__init__.py).
- **Task 4 – Forecasting:** Produced baseline and scenario forecasts for access and usage metrics using gradient boosted regressors in [notebooks/04_forecasting.ipynb](notebooks/04_forecasting.ipynb) and serialized artifacts to [models](models).
- **Dashboard Delivery:** Published interactive exploration experience through [dashboard/app.py](dashboard/app.py) to surface enriched data, events, and projections for stakeholders.

## Documented Contributions
- Authored enrichment entries dated 2026-01-28 in [data_enrichment_log.md](data_enrichment_log.md), including source citations, rationales, and confidence scoring for each manual data point.
- Designed the notebook workflow enumerated above, ensuring each task aligns with the challenge rubric and references reproducible code cells.
- Implemented modular utilities within [src/event_effects.py](src/event_effects.py) to avoid duplicated notebook logic and to promote testable analytics.
- Captured generated artifacts within [reports/figures](reports/figures) and [models](models) folders, with provenance notes embedded in notebook markdown cells.

## Repository Structure
```
ethiopia-fi-forecast/
├── .github/workflows/       # CI definitions
├── data/
│   ├── raw/                 # Provided + newly collected sources (read-only)
│   └── processed/           # Analysis-ready extracts (gitignored)
├── notebooks/               # Jupyter notebooks for Tasks 1-4
├── src/                     # Reusable data + modeling utilities
├── dashboard/               # Streamlit application
├── tests/                   # Unit and regression tests
├── models/                  # Serialized model artifacts
├── reports/figures/         # Generated visuals for documentation
├── requirements.txt         # Python dependencies
└── README.md                # You are here
```

## Getting Started
1. **Create a virtual environment** (recommended: Python 3.10+).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables** (if any) in a `.env` file at the project root. Sensitive credentials must never be committed.
4. **Run notebooks** in order:
   - Task 1: `notebooks/01_data_enrichment.ipynb`
   - Task 2: `notebooks/02_eda.ipynb`
   - Task 3: `notebooks/03_event_impact_modeling.ipynb`
   - Task 4: `notebooks/04_forecasting.ipynb`
5. **Launch dashboard** once forecasts are ready:
   ```bash
   streamlit run dashboard/app.py
   ```

## Data Handling Principles
- Treat everything under `data/raw/` as immutable snapshots of source files. Add new data by appending files with clear provenance metadata.
- Write intermediate outputs to `data/processed/` and avoid committing large derived artifacts.
- Use `data_enrichment_log.md` to document every manual addition (source URL, quote, reasoning, confidence).

## Quality and Collaboration Workflow
1. Create feature branches per task (`task-1`, `task-2`, etc.).
2. Commit logically grouped changes with descriptive messages.
3. Open Pull Requests to `main` after local linting and tests (`pytest`).
4. Ensure notebooks are cleared of extraneous outputs before committing when feasible.

## Code & Repository Best Practices Checklist
- Lint notebooks and modules with `ruff` and run `pytest` to maintain style and correctness before every commit.
- Preserve notebook outputs that serve as evidence for rubric scoring while trimming verbose intermediary cells.
- Track major decisions inside [notebooks/README.md](notebooks/README.md) and synchronize updates with the root README to keep contributors aligned.
- Gate merges behind pull requests to maintain review history documenting individual contributions.

## License & Attribution
This project is for educational purposes within the 10 Academy program. Cite original data sources (World Bank Global Findex, IMF FAS, GSMA, National Bank of Ethiopia, etc.) wherever data is used or derived.
