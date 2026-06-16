"""
forecaster.py
─────────────
AI forecasting engine for SME cash flow prediction.

Pipeline
────────
1. Aggregate raw transactions into a weekly balance series.
2. Fit separate Prophet models for inflows and outflows.
3. Combine forecasts into a net cash-flow projection.
4. Detect balance gaps below the safe threshold.
5. Generate actionable financing recommendations.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:                          # graceful fallback
    PROPHET_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _weeks_between(df: pd.DataFrame) -> float:
    span = (df["date"].max() - df["date"].min()).days
    return max(span / 7, 1)


def _build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to weekly inflow / outflow / balance."""
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").dt.to_timestamp()

    inflow  = (df[df["type"] == "inflow"]
               .groupby("week")["amount"].sum()
               .rename("inflow"))
    outflow = (df[df["type"] == "outflow"]
               .groupby("week")["amount"].sum()
               .rename("outflow"))

    weekly = pd.concat([inflow, outflow], axis=1).fillna(0)
    weekly.index = pd.to_datetime(weekly.index)

    # Fill missing weeks
    full_idx = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
    weekly = weekly.reindex(full_idx, fill_value=0)
    weekly.index.name = "ds"
    weekly = weekly.reset_index()

    weekly["net"]     = weekly["inflow"] - weekly["outflow"]
    weekly["balance"] = weekly["net"].cumsum()
    return weekly


# ── fallback model (if Prophet not installed) ─────────────────────────────────

def _simple_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """Exponential weighted moving average – used when Prophet is unavailable."""
    ewm = series.ewm(span=6, adjust=False).mean()
    last = ewm.iloc[-1]
    noise_std = series.diff().dropna().std() * 0.4
    rng = np.random.default_rng(0)
    preds = last + rng.normal(0, noise_std, horizon)
    return pd.Series(preds)


def _simple_forecast_with_bounds(series: pd.Series, horizon: int
                                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    yhat = _simple_forecast(series, horizon)
    sigma = series.std() * 0.5
    return yhat, yhat - sigma, yhat + sigma


# ── Prophet wrapper ───────────────────────────────────────────────────────────

def _prophet_forecast(train: pd.DataFrame, horizon: int,
                      yearly: bool = True) -> pd.DataFrame:
    """
    Fit Prophet and return a DataFrame with [ds, yhat, yhat_lower, yhat_upper].
    train must have columns [ds, y].
    """
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        interval_width=0.80,
        uncertainty_samples=200,
    )
    # Add monthly seasonality
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train)

    future = m.make_future_dataframe(periods=horizon, freq="W")
    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).reset_index(drop=True)


# ── main class ────────────────────────────────────────────────────────────────

class CashFlowForecaster:
    """
    Parameters
    ──────────
    df                  : raw transaction DataFrame with columns
                          [date, amount, type, category]
    forecast_weeks      : how many weeks ahead to predict
    min_safe_balance    : alert threshold for balance
    financing_lead_days : days needed to secure financing
    currency            : display currency code
    """

    def __init__(
        self,
        df: pd.DataFrame,
        forecast_weeks: int = 8,
        min_safe_balance: float = 5_000,
        financing_lead_days: int = 14,
        currency: str = "USD",
    ):
        self.df                  = df.copy()
        self.forecast_weeks      = forecast_weeks
        self.min_safe_balance    = min_safe_balance
        self.financing_lead_days = financing_lead_days
        self.currency            = currency

        span_days = (df["date"].max() - df["date"].min()).days
        self.months_of_data = span_days / 30.44

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        weekly = _build_weekly(self.df)
        metrics = self._summary_metrics(weekly)
        fcast   = self._forecast(weekly)
        gaps    = self._detect_gaps(fcast)
        recs    = self._financing_recommendations(gaps, metrics)

        return {
            "historical_weekly":      weekly.rename(columns={"week": "ds"})
                                            if "week" in weekly.columns else weekly,
            "forecast":               fcast,
            "gap_alerts":             gaps,
            "financing_recommendations": recs,
            **metrics,
        }

    # ── private: summary metrics ──────────────────────────────────────────────

    def _summary_metrics(self, weekly: pd.DataFrame) -> dict[str, float]:
        # Last 3 months ≈ 13 weeks
        recent = weekly.tail(13)
        avg_in  = recent["inflow"].mean()  * 4   # annualise to monthly
        avg_out = recent["outflow"].mean() * 4
        net     = avg_in - avg_out

        # Runway: how many weeks until balance hits zero at current burn
        burn_rate = recent["outflow"].mean() - recent["inflow"].mean()
        if burn_rate > 0:
            runway = int(weekly["balance"].iloc[-1] / burn_rate)
        else:
            runway = 999   # not burning

        return {
            "avg_monthly_inflow":  round(avg_in,  2),
            "avg_monthly_outflow": round(avg_out, 2),
            "avg_monthly_net":     round(net,     2),
            "cash_runway_weeks":   max(runway, 0),
        }

    # ── private: forecasting ──────────────────────────────────────────────────

    def _forecast(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast inflow and outflow separately, combine into balance projection.
        Falls back to EWM if Prophet is not installed.
        """
        last_balance = weekly["balance"].iloc[-1]
        last_date    = pd.to_datetime(weekly["ds"].iloc[-1]
                                      if "ds" in weekly.columns
                                      else weekly.index[-1])

        future_dates = pd.date_range(
            last_date + timedelta(weeks=1),
            periods=self.forecast_weeks,
            freq="W-MON",
        )

        if PROPHET_AVAILABLE and len(weekly) >= 8:
            in_train  = weekly[["ds", "inflow"]].rename(columns={"inflow":  "y"})
            out_train = weekly[["ds", "outflow"]].rename(columns={"outflow": "y"})

            # Prophet needs ≥ 2 non-zero data points for yearly seasonality
            yearly = self.months_of_data >= 12

            in_fc  = _prophet_forecast(in_train,  self.forecast_weeks, yearly)
            out_fc = _prophet_forecast(out_train, self.forecast_weeks, yearly)

            in_fc[["yhat", "yhat_lower", "yhat_upper"]] = \
                in_fc[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)
            out_fc[["yhat", "yhat_lower", "yhat_upper"]] = \
                out_fc[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)

            net_yhat  = in_fc["yhat"]  - out_fc["yhat"]
            net_lower = in_fc["yhat_lower"] - out_fc["yhat_upper"]   # worst case
            net_upper = in_fc["yhat_upper"] - out_fc["yhat_lower"]   # best case

        else:
            # Fallback
            in_yhat, in_lo, in_hi   = _simple_forecast_with_bounds(
                weekly["inflow"],  self.forecast_weeks)
            out_yhat, out_lo, out_hi = _simple_forecast_with_bounds(
                weekly["outflow"], self.forecast_weeks)

            in_yhat  = in_yhat.clip(lower=0)
            out_yhat = out_yhat.clip(lower=0)

            net_yhat  = in_yhat  - out_yhat
            net_lower = in_lo    - out_hi
            net_upper = in_hi    - out_lo

        # Cumulative balance from last known point
        bal_yhat  = last_balance + net_yhat.cumsum().values
        bal_lower = last_balance + net_lower.cumsum().values
        bal_upper = last_balance + net_upper.cumsum().values

        fcast = pd.DataFrame({
            "ds":         future_dates,
            "yhat":       bal_yhat,
            "yhat_lower": bal_lower,
            "yhat_upper": bal_upper,
            "net":        net_yhat.values,
        })
        return fcast

    # ── private: gap detection ────────────────────────────────────────────────

    def _detect_gaps(self, fcast: pd.DataFrame) -> list[dict]:
        gaps = []
        for _, row in fcast.iterrows():
            if row["yhat"] < self.min_safe_balance:
                gap_date  = row["ds"].date()
                apply_by  = (row["ds"] - timedelta(days=self.financing_lead_days)).date()
                shortfall = self.min_safe_balance - row["yhat"]
                severity  = "critical" if row["yhat"] < 0 else "warning"
                gaps.append({
                    "date":              str(gap_date),
                    "projected_balance": round(row["yhat"],  2),
                    "shortfall":         round(shortfall,    2),
                    "apply_by":          str(apply_by),
                    "severity":          severity,
                })
        return gaps

    # ── private: financing recommendations ───────────────────────────────────

    def _financing_recommendations(
        self,
        gaps: list[dict],
        metrics: dict,
    ) -> list[dict]:
        recs = []

        if not gaps:
            return recs

        # Sort by earliest gap
        gaps_sorted = sorted(gaps, key=lambda g: g["date"])
        worst       = max(gaps, key=lambda g: g["shortfall"])

        # Buffer: recommend 20% more than the shortfall to be safe
        amount_needed = round(worst["shortfall"] * 1.20 / 500) * 500  # round to $500

        # Urgency: how many days until the first gap?
        first_gap_date = datetime.strptime(gaps_sorted[0]["date"], "%Y-%m-%d").date()
        days_until_gap = (first_gap_date - datetime.today().date()).days

        if days_until_gap <= self.financing_lead_days:
            urgency = "High"
        elif days_until_gap <= self.financing_lead_days * 2:
            urgency = "Medium"
        else:
            urgency = "Low"

        # Revolving credit line for short gaps
        if worst["shortfall"] < 50_000:
            recs.append({
                "type":             "Business Line of Credit",
                "urgency":          urgency,
                "suggested_amount": amount_needed,
                "apply_by":         gaps_sorted[0]["apply_by"],
                "reason": (
                    f"A revolving credit facility covers the projected "
                    f"shortfall of approx. {self._fmt(worst['shortfall'])} "
                    f"around {worst['date']}. Draw only what you need "
                    f"and repay as inflows recover — minimising interest."
                ),
            })

        # Invoice financing if outflow > inflow (AR lag)
        if metrics["avg_monthly_outflow"] > metrics["avg_monthly_inflow"] * 0.85:
            recs.append({
                "type":             "Invoice Financing / Factoring",
                "urgency":          "Medium",
                "suggested_amount": round(metrics["avg_monthly_inflow"] * 0.40 / 500) * 500,
                "apply_by":         gaps_sorted[0]["apply_by"],
                "reason": (
                    "Your outflow closely tracks inflow, suggesting a receivables "
                    "lag. Factoring outstanding invoices can accelerate cash "
                    "receipt by 30–60 days and prevent recurring gaps."
                ),
            })

        # Term loan for large / persistent gaps
        if worst["shortfall"] >= 50_000 or len(gaps) > self.forecast_weeks // 2:
            recs.append({
                "type":             "Short-term Business Loan",
                "urgency":          urgency,
                "suggested_amount": amount_needed,
                "apply_by":         gaps_sorted[0]["apply_by"],
                "reason": (
                    "The gap appears persistent or significant. A fixed-term "
                    "loan provides a lump sum to stabilise operations while "
                    "you grow revenue or reduce costs structurally."
                ),
            })

        # Deduplicate and sort by urgency
        priority = {"High": 0, "Medium": 1, "Low": 2}
        recs = sorted(recs, key=lambda r: priority[r["urgency"]])
        return recs

    def _fmt(self, val: float) -> str:
        symbols = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "₨", "AED": "د.إ"}
        s = symbols.get(self.currency, self.currency + " ")
        if abs(val) >= 1_000:
            return f"{s}{val/1_000:,.1f}K"
        return f"{s}{val:,.0f}"

