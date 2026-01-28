"""Streamlit dashboard entry point for the Ethiopia Financial Inclusion Forecasting project."""

import pathlib
from typing import Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data(show_spinner=False)
def load_core_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load unified data and reference codes if available."""
    data_path = RAW_DATA_DIR / "ethiopia_fi_unified_data.csv"
    ref_path = RAW_DATA_DIR / "reference_codes.csv"

    df_data = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()
    df_ref = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()
    return df_data, df_ref


def main() -> None:
    st.set_page_config(
        page_title="Ethiopia Financial Inclusion Tracker",
        page_icon="📈",
        layout="wide",
    )

    st.title("Ethiopia Financial Inclusion Tracker")
    st.caption("Selam Analytics • Week 10 Challenge • 10 Academy")

    df_data, df_ref = load_core_datasets()

    if df_data.empty:
        st.warning("Upload or generate the unified dataset (.csv) under data/raw to populate the dashboard.")
        return

    st.subheader("Quick Peek")
    st.write(df_data.head())


if __name__ == "__main__":
    main()
