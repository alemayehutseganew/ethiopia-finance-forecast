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

## License & Attribution
This project is for educational purposes within the 10 Academy program. Cite original data sources (World Bank Global Findex, IMF FAS, GSMA, National Bank of Ethiopia, etc.) wherever data is used or derived.
