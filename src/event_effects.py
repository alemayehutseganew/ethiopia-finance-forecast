"""Utilities for translating event impact links into time-distributed indicator effects."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def to_month_start(value: object) -> pd.Timestamp | pd.NaT:
    """Normalize arbitrary date-like inputs to month-start timestamps."""
    if pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value).to_period("M").to_timestamp()


def build_effect_curve(
    row: pd.Series,
    timeline: Iterable[pd.Timestamp],
    ramp_months: int = 6,
    persistence_months: int = 18,
    decay_months: int = 12,
) -> pd.Series:
    """Return a ramp–plateau–decay effect curve for a single event."""
    timeline_index = pd.Index(timeline)
    effect = pd.Series(0.0, index=timeline_index)
    impact_value = row.get("impact_numeric", 0.0)
    if pd.isna(impact_value) or impact_value == 0 or pd.isna(row.get("event_date")):
        return effect

    start_date = to_month_start(row["event_date"]) + pd.DateOffset(months=int(row.get("lag_months", 0)))
    if pd.isna(start_date) or start_date > timeline_index[-1]:
        return effect

    start_idx = effect.index.searchsorted(start_date)
    if start_idx >= len(effect):
        return effect

    ramp_len = max(1, int(ramp_months))
    persistence_len = max(0, int(persistence_months))
    decay_len = max(0, int(decay_months))

    ramp_end = min(len(effect), start_idx + ramp_len)
    plateau_end = min(len(effect), ramp_end + persistence_len)
    decay_end = min(len(effect), plateau_end + decay_len)

    ramp_span = ramp_end - start_idx
    if ramp_span > 0:
        effect.iloc[start_idx:ramp_end] = np.linspace(0, impact_value, ramp_span, endpoint=False)

    plateau_span = plateau_end - ramp_end
    if plateau_span > 0:
        effect.iloc[ramp_end:plateau_end] = impact_value

    decay_span = decay_end - plateau_end
    if decay_span > 0:
        effect.iloc[plateau_end:decay_end] = np.linspace(impact_value, 0, decay_span, endpoint=False)

    return effect


def simulate_indicator_series(
    impact_df: pd.DataFrame,
    indicator_code: str,
    start: str = "2018-01-01",
    end: str = "2028-12-01",
    ramp_months: int = 6,
    persistence_months: int = 18,
    decay_months: int = 12,
) -> pd.DataFrame:
    """Aggregate event effect curves for a given indicator across the provided timeline."""
    timeline = pd.date_range(start, end, freq="MS")
    subset = impact_df.loc[impact_df["related_indicator"] == indicator_code]
    effect = pd.Series(0.0, index=timeline)

    for _, row in subset.iterrows():
        event_curve = build_effect_curve(
            row,
            timeline,
            ramp_months=ramp_months,
            persistence_months=persistence_months,
            decay_months=decay_months,
        )
        effect = effect.add(event_curve, fill_value=0.0)

    return (
        effect.to_frame(name="modeled_effect_pp")
        .reset_index()
        .rename(columns={"index": "date"})
        .assign(indicator_code=indicator_code)
    )
