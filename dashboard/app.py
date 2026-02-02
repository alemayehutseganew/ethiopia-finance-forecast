"""Streamlit dashboard for the Ethiopia Financial Inclusion Forecasting project."""

from __future__ import annotations

import io
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.event_effects import simulate_indicator_series

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMPACT_LINKS_PATH = RAW_DATA_DIR / "impact_links.csv"
TARGET_ACCESS = 60.0


@dataclass
class ForecastResult:
    indicator: str
    base: pd.DataFrame
    scenarios_long: pd.DataFrame


@st.cache_data(show_spinner=False)
def load_core_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load unified records and impact links from disk."""
    data_path = RAW_DATA_DIR / "ethiopia_fi_unified_data.csv"
    if not data_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    records = pd.read_csv(
        data_path,
        parse_dates=["observation_date", "period_start", "period_end"],
        infer_datetime_format=True,
    )
    impact_links = pd.read_csv(IMPACT_LINKS_PATH) if IMPACT_LINKS_PATH.exists() else pd.DataFrame()
    return records, impact_links


def prepare_frames(records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = (
        records.loc[records["record_type"] == "observation"].copy()
    )
    observations["value"] = observations["value_numeric"]
    observations["observation_date"] = pd.to_datetime(observations["observation_date"], errors="coerce")
    observations["year"] = observations["observation_date"].dt.year

    events = records.loc[records["record_type"] == "event"].copy()
    events["event_date"] = pd.to_datetime(events["observation_date"], errors="coerce")

    targets = records.loc[records["record_type"] == "target"].copy()
    return observations, events, targets


def build_indicator_meta(observations: pd.DataFrame) -> pd.DataFrame:
    return (
        observations[["indicator_code", "indicator", "pillar"]]
        .dropna(subset=["indicator_code"])
        .drop_duplicates()
        .rename(columns={"indicator": "indicator_name", "pillar": "indicator_theme"})
    )


def enrich_impact_links(
    impact_links: pd.DataFrame,
    events: pd.DataFrame,
    indicator_meta: pd.DataFrame,
) -> pd.DataFrame:
    if impact_links.empty:
        return pd.DataFrame(columns=[
            "record_id",
            "related_indicator",
            "impact_numeric",
            "lag_months",
            "event_date",
        ])

    impact = impact_links.rename(columns={"pillar": "impact_pillar", "indicator": "link_label"}).copy()
    impact["impact_estimate"] = pd.to_numeric(impact.get("impact_estimate"), errors="coerce")
    impact["lag_months"] = pd.to_numeric(impact.get("lag_months"), errors="coerce").fillna(0).astype(int)
    impact["impact_direction"] = impact.get("impact_direction", "increase").str.lower()
    impact["impact_magnitude"] = impact.get("impact_magnitude", "medium").str.lower()

    magnitude_defaults = {"low": 2.5, "medium": 5.0, "high": 10.0}
    direction_sign = impact["impact_direction"].map({"increase": 1, "decrease": -1, "mixed": 0}).fillna(1)
    impact_numeric = impact["impact_estimate"].where(
        impact["impact_estimate"].notna(),
        impact["impact_magnitude"].map(magnitude_defaults),
    ).fillna(0.0)
    impact["impact_numeric"] = impact_numeric.astype(float) * direction_sign

    events_lookup = (
        events.rename(
            columns={
                "record_id": "event_id",
                "indicator": "event_name",
                "category": "event_category",
                "observation_date": "raw_event_date",
            }
        )
        .assign(event_date=lambda df: pd.to_datetime(df["raw_event_date"], errors="coerce"))
        [["event_id", "event_name", "event_category", "event_date", "source_name", "source_url", "notes"]]
    )

    impact = impact.merge(events_lookup, left_on="parent_id", right_on="event_id", how="left")
    impact = impact.merge(
        indicator_meta.rename(
            columns={
                "indicator_code": "related_indicator",
                "indicator_name": "target_indicator_name",
                "indicator_theme": "target_theme",
            }
        ),
        on="related_indicator",
        how="left",
    )
    impact["event_date"] = pd.to_datetime(impact["event_date"], errors="coerce")
    return impact


def latest_value(observations: pd.DataFrame, indicator_code: str) -> Tuple[float | None, int | None]:
    subset = (
        observations.loc[observations["indicator_code"] == indicator_code]
        .dropna(subset=["observation_date", "value_numeric"])
        .sort_values("observation_date")
    )
    if subset.empty:
        return None, None
    last = subset.iloc[-1]
    return float(last["value_numeric"]), int(last["year"])


def growth_rate(observations: pd.DataFrame, indicator_code: str) -> float | None:
    subset = (
        observations.loc[observations["indicator_code"] == indicator_code]
        .dropna(subset=["year", "value_numeric"])
        .sort_values("year")
    )
    if len(subset) < 2:
        return None
    latest = subset.iloc[-1]["value_numeric"]
    prev = subset.iloc[-2]["value_numeric"]
    if prev == 0:
        return None
    return (latest - prev) / prev * 100


def format_metric(value: float | None, suffix: str = "%") -> str:
    if value is None or np.isnan(value):
        return "–"
    return f"{value:,.2f}{suffix}"


def regression_forecast(
    indicator_code: str,
    observations: pd.DataFrame,
    impact_enriched: pd.DataFrame,
    years_ahead: Iterable[int] = (2025, 2026, 2027),
) -> ForecastResult | None:
    series = (
        observations.loc[observations["indicator_code"] == indicator_code, ["year", "value_numeric"]]
        .dropna()
        .drop_duplicates(subset=["year"], keep="last")
        .sort_values("year")
    )
    if len(series) < 2:
        return None

    years = series["year"].to_numpy(dtype=float)
    values = series["value_numeric"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(years, values, 1)
    fitted = intercept + slope * years
    residuals = values - fitted
    if len(series) > 2:
        se = np.sqrt(np.sum(residuals ** 2) / (len(series) - 2))
    else:
        se = np.std(residuals) if np.std(residuals) > 0 else 5.0
    x_mean = years.mean()
    sxx = np.sum((years - x_mean) ** 2)

    future_years = np.array(list(years_ahead), dtype=float)
    baseline = intercept + slope * future_years
    if sxx == 0:
        se_mean = se * np.sqrt(1 / len(series))
    else:
        se_mean = se * np.sqrt(1 / len(series) + (future_years - x_mean) ** 2 / sxx)
    margin = 1.96 * se_mean
    ci_lower = baseline - margin
    ci_upper = baseline + margin

    min_year = int(min(years.min(), future_years.min()) - 5)
    event_effect_df = simulate_indicator_series(
        impact_enriched,
        indicator_code=indicator_code,
        start=f"{min_year}-01-01",
        end=f"{int(future_years.max())}-12-01",
        ramp_months=6,
        persistence_months=24,
        decay_months=12,
    )
    event_yearly = (
        event_effect_df.assign(year=lambda df: df["date"].dt.year)
        .groupby("year")["modeled_effect_pp"]
        .sum()
    )

    baseline_df = pd.DataFrame(
        {
            "year": future_years.astype(int),
            "baseline": baseline,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "event_effect": event_yearly.reindex(future_years.astype(int), fill_value=0.0).values,
        }
    )

    last_year = int(series.iloc[-1]["year"])
    last_value = float(series.iloc[-1]["value_numeric"])

    def scenario_value(row: pd.Series, growth_mult: float, event_mult: float) -> float:
        incremental = row["baseline"] - last_value
        return last_value + incremental * growth_mult + row["event_effect"] * event_mult

    baseline_df = baseline_df.assign(
        scenario_baseline=lambda df: df["baseline"],
        scenario_event=lambda df: df.apply(scenario_value, axis=1, args=(1.0, 1.0)),
        scenario_optimistic=lambda df: df.apply(scenario_value, axis=1, args=(1.25, 1.6)),
        scenario_pessimistic=lambda df: df.apply(scenario_value, axis=1, args=(0.7, 0.5)),
        event_ci_lower=lambda df: df["ci_lower"] + df["event_effect"],
        event_ci_upper=lambda df: df["ci_upper"] + df["event_effect"],
    )

    records: List[Dict[str, float]] = []
    for _, row in baseline_df.iterrows():
        for name, value, lower, upper in [
            ("baseline", row["scenario_baseline"], row["ci_lower"], row["ci_upper"]),
            ("event_augmented", row["scenario_event"], row["event_ci_lower"], row["event_ci_upper"]),
            ("optimistic", row["scenario_optimistic"], np.nan, np.nan),
            ("pessimistic", row["scenario_pessimistic"], np.nan, np.nan),
        ]:
            records.append(
                {
                    "year": int(row["year"]),
                    "indicator": indicator_code,
                    "scenario": name,
                    "forecast_pp": value,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )

    scenarios_long = pd.DataFrame(records)
    return ForecastResult(indicator=indicator_code, base=baseline_df, scenarios_long=scenarios_long)


def overview_page(observations: pd.DataFrame, forecasts: Dict[str, ForecastResult]) -> None:
    st.header("Overview")

    acc_value, acc_year = latest_value(observations, "ACC_OWNERSHIP")
    usage_value, usage_year = latest_value(observations, "USG_DIGITAL_PAYMENT")
    crossover_value, _ = latest_value(observations, "USG_CROSSOVER")
    acc_growth = growth_rate(observations, "ACC_OWNERSHIP")
    usage_growth = growth_rate(observations, "USG_DIGITAL_PAYMENT")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Account ownership", format_metric(acc_value), f"Year {acc_year}" if acc_year else "—")
    col2.metric("Digital payment usage", format_metric(usage_value), f"Year {usage_year}" if usage_year else "—")
    col3.metric("Account ownership YoY", format_metric(acc_growth, suffix="pp"))
    col4.metric("Digital payments YoY", format_metric(usage_growth, suffix="pp"))

    st.subheader("Account ownership trajectory")
    acc_series = (
        observations.loc[observations["indicator_code"] == "ACC_OWNERSHIP", ["observation_date", "value_numeric"]]
        .dropna()
        .sort_values("observation_date")
    )
    if not acc_series.empty:
        fig = px.line(
            acc_series,
            x="observation_date",
            y="value_numeric",
            markers=True,
            labels={"observation_date": "Date", "value_numeric": "% adults"},
        )
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    scenario_table = forecasts.get("ACC_OWNERSHIP")
    if scenario_table:
        download_dataframe("Account ownership forecast", scenario_table.scenarios_long)


def trends_page(observations: pd.DataFrame, events: pd.DataFrame) -> None:
    st.header("Trends")

    indicator_options = (
        observations[["indicator_code", "indicator", "pillar"]]
        .dropna(subset=["indicator_code"])
        .drop_duplicates()
        .sort_values(["pillar", "indicator"])
    )
    default_code = "ACC_MM_ACCOUNT" if "ACC_MM_ACCOUNT" in indicator_options["indicator_code"].values else indicator_options.iloc[0]["indicator_code"]
    selected_code = st.selectbox(
        "Choose indicator",
        options=indicator_options["indicator_code"],
        format_func=lambda code: indicator_options.set_index("indicator_code").loc[code, "indicator"],
        index=indicator_options["indicator_code"].tolist().index(default_code),
    )

    subset = (
        observations.loc[observations["indicator_code"] == selected_code, ["observation_date", "value_numeric", "source_name"]]
        .dropna()
        .sort_values("observation_date")
    )
    if subset.empty:
        st.info("No observations available for the selected indicator.")
    else:
        min_date = subset["observation_date"].min()
        max_date = subset["observation_date"].max()
        date_range = st.slider(
            "Select date range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        )
        mask = subset["observation_date"].between(*date_range)
        filtered = subset.loc[mask]
        fig = px.line(
            filtered,
            x="observation_date",
            y="value_numeric",
            markers=True,
            hover_data=["source_name"],
            labels={"observation_date": "Date", "value_numeric": "Value"},
        )
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel comparison: P2P vs ATM")
    channel_codes = ["USG_P2P_COUNT", "USG_ATM_COUNT", "USG_P2P_VALUE", "USG_ATM_VALUE"]
    channel_subset = (
        observations.loc[observations["indicator_code"].isin(channel_codes), ["indicator_code", "year", "value_numeric"]]
        .dropna()
        .pivot_table(index="year", columns="indicator_code", values="value_numeric")
    )
    if not channel_subset.empty:
        fig_compare = go.Figure()
        if "USG_P2P_COUNT" in channel_subset.columns and "USG_ATM_COUNT" in channel_subset.columns:
            fig_compare.add_trace(
                go.Bar(name="P2P transactions", x=channel_subset.index, y=channel_subset["USG_P2P_COUNT"], offsetgroup=0, marker_color="#2a9d8f")
            )
            fig_compare.add_trace(
                go.Bar(name="ATM transactions", x=channel_subset.index, y=channel_subset["USG_ATM_COUNT"], offsetgroup=1, marker_color="#e76f51")
            )
        fig_compare.update_layout(
            barmode="group",
            height=420,
            template="plotly_white",
            xaxis_title="Fiscal year",
            yaxis_title="Transactions",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Digital finance event timeline")
    events_subset = events.dropna(subset=["event_date", "indicator"])
    if not events_subset.empty:
        fig_events = px.scatter(
            events_subset,
            x="event_date",
            y="category",
            color="category",
            hover_data=["indicator", "value_text", "notes"],
        )
        fig_events.update_traces(marker=dict(size=12))
        fig_events.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig_events, use_container_width=True)


def forecasts_page(forecasts: Dict[str, ForecastResult], observations: pd.DataFrame) -> None:
    st.header("Forecasts")

    if not forecasts:
        st.info("Forecasts require at least two historical observations and available impact links.")
        return

    indicator_map = {
        "ACC_OWNERSHIP": "Account ownership",
        "USG_DIGITAL_PAYMENT": "Digital payment usage",
    }
    available_codes = [code for code in indicator_map if code in forecasts]
    selected_indicator = st.selectbox(
        "Indicator",
        options=available_codes,
        format_func=lambda code: indicator_map.get(code, code),
    )

    scenario_options = ["baseline", "event_augmented", "optimistic", "pessimistic"]
    selected_scenarios = st.multiselect(
        "Scenarios to display",
        options=scenario_options,
        default=["baseline", "event_augmented"],
    )
    if not selected_scenarios:
        st.warning("Select at least one scenario to display.")
        return

    forecast_result = forecasts[selected_indicator]
    history = (
        observations.loc[observations["indicator_code"] == selected_indicator, ["year", "value_numeric"]]
        .dropna()
        .sort_values("year")
    )

    fig = go.Figure()
    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["year"],
                y=history["value_numeric"],
                mode="lines+markers",
                name="Observed",
                line=dict(color="#264653"),
            )
        )

    scenario_styles = {
        "baseline": dict(color="#2a9d8f", dash="solid"),
        "event_augmented": dict(color="#e76f51", dash="dash"),
        "optimistic": dict(color="#f4a261", dash="dot"),
        "pessimistic": dict(color="#6d597a", dash="dashdot"),
    }
    for scenario in selected_scenarios:
        scenario_slice = forecast_result.scenarios_long.loc[
            forecast_result.scenarios_long["scenario"] == scenario
        ]
        fig.add_trace(
            go.Scatter(
                x=scenario_slice["year"],
                y=scenario_slice["forecast_pp"],
                mode="lines+markers",
                name=scenario.replace("_", " ").title(),
                line=scenario_styles.get(scenario, dict()),
            )
        )

    baseline_slice = forecast_result.scenarios_long.loc[
        forecast_result.scenarios_long["scenario"] == "baseline"
    ]
    fig.add_trace(
        go.Scatter(
            x=baseline_slice["year"],
            y=baseline_slice["ci_upper"],
            mode="lines",
            line=dict(color="rgba(42, 157, 143, 0.2)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_slice["year"],
            y=baseline_slice["ci_lower"],
            mode="lines",
            line=dict(color="rgba(42, 157, 143, 0.2)"),
            fill="tonexty",
            fillcolor="rgba(42, 157, 143, 0.1)",
            name="Baseline 95% CI",
        )
    )

    fig.update_layout(
        height=460,
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title="% of adults",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        forecast_result.scenarios_long.sort_values(["scenario", "year"]).reset_index(drop=True),
        use_container_width=True,
    )
    download_dataframe("Forecast scenarios", forecast_result.scenarios_long)


def inclusion_page(forecasts: Dict[str, ForecastResult]) -> None:
    st.header("Inclusion Projections")

    if "ACC_OWNERSHIP" not in forecasts or "USG_DIGITAL_PAYMENT" not in forecasts:
        st.info("Run Task 4 enrichment to unlock scenario projections.")
        return

    scenario_map = {
        "baseline": "Baseline",
        "event_augmented": "Event-adjusted",
        "optimistic": "Optimistic",
        "pessimistic": "Pessimistic",
    }
    scenario_choice = st.selectbox("Scenario", options=list(scenario_map.keys()), format_func=lambda k: scenario_map[k])

    acc_table = forecasts["ACC_OWNERSHIP"].scenarios_long
    usage_table = forecasts["USG_DIGITAL_PAYMENT"].scenarios_long
    latest_year = int(acc_table["year"].max())
    acc_value = acc_table.loc[
        (acc_table["scenario"] == scenario_choice) & (acc_table["year"] == latest_year),
        "forecast_pp",
    ].squeeze()
    usage_value = usage_table.loc[
        (usage_table["scenario"] == scenario_choice) & (usage_table["year"] == latest_year),
        "forecast_pp",
    ].squeeze()

    col1, col2 = st.columns(2)
    col1.metric("Projected account ownership", format_metric(acc_value), f"FY {latest_year}")
    col2.metric("Projected usage", format_metric(usage_value), f"FY {latest_year}")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=acc_value,
            delta={"reference": TARGET_ACCESS, "increasing": {"color": "#2a9d8f"}},
            gauge={
                "axis": {"range": [0, max(80, TARGET_ACCESS + 10)]},
                "bar": {"color": "#2a9d8f"},
                "steps": [
                    {"range": [0, TARGET_ACCESS], "color": "#f4a261"},
                    {"range": [TARGET_ACCESS, max(80, TARGET_ACCESS + 10)], "color": "#e76f51"},
                ],
                "threshold": {"line": {"color": "#264653", "width": 3}, "value": TARGET_ACCESS},
            },
        )
    )
    gauge.update_layout(height=400, margin=dict(t=50, b=50))
    st.plotly_chart(gauge, use_container_width=True)

    combined = pd.concat(
        [
            acc_table.assign(indicator="Account ownership"),
            usage_table.assign(indicator="Digital payment usage"),
        ],
        ignore_index=True,
    )
    filtered = combined.loc[combined["scenario"] == scenario_choice]
    fig = px.line(
        filtered,
        x="year",
        y="forecast_pp",
        color="indicator",
        markers=True,
        labels={"forecast_pp": "% adults"},
    )
    fig.update_layout(height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def download_dataframe(label: str, df: pd.DataFrame) -> None:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label=f"Download {label}",
        data=buffer.getvalue(),
        file_name=f"{label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def compute_forecasts(
    observations: pd.DataFrame,
    impact_enriched: pd.DataFrame,
    indicator_codes: Iterable[str],
) -> Dict[str, ForecastResult]:
    forecasts: Dict[str, ForecastResult] = {}
    for code in indicator_codes:
        forecast = regression_forecast(code, observations, impact_enriched)
        if forecast:
            forecasts[code] = forecast
    return forecasts


def main() -> None:
    st.set_page_config(
        page_title="Ethiopia Financial Inclusion Tracker",
        page_icon="📈",
        layout="wide",
    )

    st.title("Ethiopia Financial Inclusion Tracker")
    st.caption("Selam Analytics • Week 10 Challenge • 10 Academy")

    records, impact_links = load_core_datasets()
    if records.empty:
        st.warning("Upload or generate the unified dataset (data/raw/ethiopia_fi_unified_data.csv) to run the dashboard.")
        return

    observations, events, _ = prepare_frames(records)
    indicator_meta = build_indicator_meta(observations)
    impact_enriched = enrich_impact_links(impact_links, events, indicator_meta)
    forecasts = compute_forecasts(
        observations,
        impact_enriched,
        indicator_codes=["ACC_OWNERSHIP", "USG_DIGITAL_PAYMENT"],
    )

    page = st.sidebar.radio(
        "Navigate",
        options=["Overview", "Trends", "Forecasts", "Inclusion Projections"],
    )

    download_dataframe("Unified dataset", records)

    if page == "Overview":
        overview_page(observations, forecasts)
    elif page == "Trends":
        trends_page(observations, events)
    elif page == "Forecasts":
        forecasts_page(forecasts, observations)
    else:
        inclusion_page(forecasts)


if __name__ == "__main__":
    main()
