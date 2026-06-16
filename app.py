import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings("ignore")

from forecaster import CashFlowForecaster

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Cash Flow AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0F1923;
    border-right: 1px solid #1E2D3D;
  }
  section[data-testid="stSidebar"] * { color: #C8D8E8 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

  /* Main background */
  .main { background: #F0F4F8; }
  .block-container { padding: 2rem 2.5rem 3rem; }

  /* KPI cards */
  .kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    border-left: 4px solid #2563EB;
    margin-bottom: 1rem;
  }
  .kpi-card.danger  { border-left-color: #DC2626; }
  .kpi-card.warning { border-left-color: #D97706; }
  .kpi-card.success { border-left-color: #059669; }
  .kpi-label { font-size: .75rem; font-weight: 600; letter-spacing: .08em;
               text-transform: uppercase; color: #64748B; margin-bottom: .3rem; }
  .kpi-value { font-size: 1.9rem; font-weight: 700; color: #0F172A; line-height: 1; }
  .kpi-sub   { font-size: .8rem; color: #94A3B8; margin-top: .35rem; }

  /* Alert banner */
  .alert-box {
    background: #FEF3C7; border: 1px solid #F59E0B;
    border-radius: 10px; padding: 1rem 1.4rem; margin: 1rem 0;
    display: flex; align-items: flex-start; gap: .8rem;
  }
  .alert-box.critical {
    background: #FEE2E2; border-color: #EF4444;
  }
  .alert-box.ok {
    background: #D1FAE5; border-color: #10B981;
  }

  /* Section header */
  .section-header {
    font-size: 1.05rem; font-weight: 600; color: #1E293B;
    padding-bottom: .4rem; border-bottom: 2px solid #E2E8F0;
    margin: 1.8rem 0 1rem;
  }

  /* Financing recommendation card */
  .rec-card {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: .6rem 0;
  }
  .rec-card.urgent {
    background: #FFF7ED; border-color: #FED7AA;
  }
  .rec-badge {
    display: inline-block; font-size: .7rem; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase;
    padding: .15rem .5rem; border-radius: 999px;
    background: #2563EB; color: #fff; margin-bottom: .5rem;
  }
  .rec-badge.urgent { background: #EA580C; }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    border: 2px dashed #CBD5E1; border-radius: 10px;
    background: #F8FAFC;
  }

  /* Plotly chart border */
  [data-testid="stPlotlyChart"] {
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }

  /* Metric delta colours */
  [data-testid="stMetricDelta"] svg { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_currency(val: float, currency: str = "USD") -> str:
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "₨", "AED": "د.إ"}
    s = symbols.get(currency, currency + " ")
    if abs(val) >= 1_000_000:
        return f"{s}{val/1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"{s}{val/1_000:,.1f}K"
    return f"{s}{val:,.0f}"


def kpi(label, value, sub="", card_class=""):
    return f"""
    <div class="kpi-card {card_class}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>"""


def alert(msg, level="warning"):
    icons = {"warning": "⚠️", "critical": "🔴", "ok": "✅"}
    return f'<div class="alert-box {level}">{icons[level]} {msg}</div>'


def generate_sample_data() -> pd.DataFrame:
    """Create 6 months of realistic SME transaction data."""
    rng = np.random.default_rng(42)
    today = datetime.today()
    start = today - timedelta(days=180)
    dates = pd.date_range(start, today, freq="D")

    records = []
    for d in dates:
        # Revenue: weekdays only, seasonality bump mid-month
        if d.weekday() < 5:
            base_rev = 3_500
            seasonal = 1.15 if 10 <= d.day <= 20 else 1.0
            n_txns = rng.integers(1, 5)
            for _ in range(n_txns):
                amt = rng.normal(base_rev * seasonal / n_txns, 600)
                records.append({"date": d.date(), "amount": max(amt, 50),
                                 "type": "inflow", "category": "Sales Revenue"})

        # Fixed costs: rent on 1st, payroll on 15th
        if d.day == 1:
            records.append({"date": d.date(), "amount": 4_200,
                             "type": "outflow", "category": "Rent"})
        if d.day == 15:
            records.append({"date": d.date(), "amount": 12_000,
                             "type": "outflow", "category": "Payroll"})

        # Variable costs: random utilities / supplies
        if rng.random() < 0.35:
            amt = rng.choice([300, 550, 900, 1_200, 2_000],
                             p=[0.4, 0.25, 0.2, 0.1, 0.05])
            cat = rng.choice(["Utilities", "Supplies", "Marketing",
                               "Software", "Logistics"])
            records.append({"date": d.date(), "amount": float(amt),
                             "type": "outflow", "category": cat})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_uploaded(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip().str.lower()

    # Flexible column mapping
    col_map = {}
    for c in df.columns:
        if any(x in c for x in ["date", "time", "when"]):
            col_map["date"] = c
        elif any(x in c for x in ["amount", "value", "sum", "total", "amt"]):
            col_map["amount"] = c
        elif any(x in c for x in ["type", "direction", "flow", "kind"]):
            col_map["type"] = c
        elif any(x in c for x in ["category", "cat", "description", "desc", "note"]):
            col_map["category"] = c

    required = {"date", "amount", "type"}
    missing = required - set(col_map.keys())
    if missing:
        st.error(f"Could not find columns for: {', '.join(missing)}. "
                 "Your file needs date, amount, and type (inflow/outflow) columns.")
        st.stop()

    df = df.rename(columns={v: k for k, v in col_map.items()})
    if "category" not in df.columns:
        df["category"] = "General"

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df = df.dropna(subset=["date", "amount"])

    # Normalise type
    df["type"] = df["type"].str.lower().str.strip()
    df["type"] = df["type"].replace({
        "credit": "inflow", "debit": "outflow",
        "in": "inflow", "out": "outflow",
        "income": "inflow", "expense": "outflow",
        "revenue": "inflow", "cost": "outflow",
    })
    return df[["date", "amount", "type", "category"]].sort_values("date").reset_index(drop=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 SME Cash Flow AI")
    st.markdown("*AI-powered forecasting & financing alerts*")
    st.divider()

    st.markdown("### ⚙️ Settings")
    currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "PKR", "AED"], index=0)
    forecast_horizon = st.slider("Forecast horizon (weeks)", 4, 16, 8)
    min_balance = st.number_input(
        "Minimum safe balance", value=5_000, step=500,
        help="Alert when projected balance drops below this threshold."
    )
    lead_days = st.slider(
        "Financing lead time (days)", 7, 45, 14,
        help="Days needed to secure financing before a gap hits."
    )

    st.divider()
    st.markdown("### 📂 Transaction Data")
    data_source = st.radio("Data source", ["Use sample data", "Upload my data"])

    uploaded = None
    if data_source == "Upload my data":
        uploaded = st.file_uploader(
            "CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Needs: date, amount, type (inflow/outflow), category (optional)"
        )
        with st.expander("Required format"):
            st.dataframe(pd.DataFrame({
                "date": ["2024-01-03", "2024-01-04"],
                "amount": [3500, 1200],
                "type": ["inflow", "outflow"],
                "category": ["Sales", "Rent"],
            }), use_container_width=True)

    st.divider()
    st.caption("Built with Prophet · Streamlit · Plotly")


# ── Data loading ──────────────────────────────────────────────────────────────
if data_source == "Use sample data":
    df = generate_sample_data()
    st.info("📊 Showing **sample data** (6 months, simulated SME). "
            "Upload your own data in the sidebar for real insights.", icon="ℹ️")
else:
    if uploaded is None:
        st.warning("⬆️ Upload a transaction file to get started.")
        st.stop()
    df = load_uploaded(uploaded)


# ── Run forecaster ────────────────────────────────────────────────────────────
with st.spinner("🤖 Running AI forecast…"):
    fc = CashFlowForecaster(
        df,
        forecast_weeks=forecast_horizon,
        min_safe_balance=min_balance,
        financing_lead_days=lead_days,
        currency=currency,
    )
    results = fc.run()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# SME Cash Flow Forecasting")
months_label = f"{fc.months_of_data:.1f}" if hasattr(fc, "months_of_data") else "?"
st.markdown(
    f"Analysing **{len(df):,} transactions** across "
    f"**{months_label} months** · "
    f"Forecasting **{forecast_horizon} weeks** ahead"
)

# ── Alert banner ──────────────────────────────────────────────────────────────
gaps = results["gap_alerts"]
if gaps:
    first_gap = gaps[0]
    if first_gap["severity"] == "critical":
        st.markdown(
            alert(
                f"🚨 <strong>Critical cash gap detected</strong> — "
                f"balance projected to hit "
                f"{fmt_currency(first_gap['projected_balance'], currency)} "
                f"around {first_gap['date']}. "
                f"Apply for financing by <strong>{first_gap['apply_by']}</strong>.",
                "critical"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            alert(
                f"Potential cash gap around <strong>{first_gap['date']}</strong>. "
                f"Consider arranging financing by {first_gap['apply_by']}."
            ),
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        alert("Cash flow looks healthy over the forecast period. No gaps detected.", "ok"),
        unsafe_allow_html=True,
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        kpi("Avg Monthly Inflow",
            fmt_currency(results["avg_monthly_inflow"], currency),
            "Last 3 months average"),
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        kpi("Avg Monthly Outflow",
            fmt_currency(results["avg_monthly_outflow"], currency),
            "Last 3 months average",
            "warning" if results["avg_monthly_outflow"] > results["avg_monthly_inflow"] else ""),
        unsafe_allow_html=True
    )
with c3:
    net = results["avg_monthly_net"]
    st.markdown(
        kpi("Avg Net Cash Flow",
            fmt_currency(net, currency),
            "Inflow minus outflow",
            "danger" if net < 0 else "success"),
        unsafe_allow_html=True
    )
with c4:
    runway = results["cash_runway_weeks"]
    rw_label = f"{runway} weeks" if runway < 52 else "12+ months"
    st.markdown(
        kpi("Cash Runway",
            rw_label,
            f"At current burn rate",
            "danger" if runway < 8 else ("warning" if runway < 16 else "success")),
        unsafe_allow_html=True
    )

# ── Forecast chart ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Cash Flow Forecast</div>', unsafe_allow_html=True)

hist = results["historical_weekly"]
fcast = results["forecast"]

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.06,
    subplot_titles=("Weekly Cash Balance (Historical + Forecast)", "Weekly Net Cash Flow"),
)

# Balance – historical
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["balance"],
    name="Historical Balance",
    line=dict(color="#2563EB", width=2.5),
    mode="lines",
), row=1, col=1)

# Balance – forecast with confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([fcast["ds"], fcast["ds"].iloc[::-1]]),
    y=pd.concat([fcast["yhat_upper"], fcast["yhat_lower"].iloc[::-1]]),
    fill="toself", fillcolor="rgba(37,99,235,0.10)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Band", showlegend=True,
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=fcast["ds"], y=fcast["yhat"],
    name="Forecast Balance",
    line=dict(color="#2563EB", width=2.5, dash="dash"),
    mode="lines",
), row=1, col=1)

# Minimum safe balance line
fig.add_hline(
    y=min_balance, row=1, col=1,
    line=dict(color="#DC2626", width=1.5, dash="dot"),
    annotation_text=f"Min Safe ({fmt_currency(min_balance, currency)})",
    annotation_position="top left",
    annotation_font_color="#DC2626",
)

# Gap markers
for g in gaps:
    fig.add_vline(
        x=g["date"], row=1, col=1,
        line=dict(color="#DC2626", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=g["date"], y=g["projected_balance"],
        text="⚠ Gap", showarrow=True, arrowhead=2,
        arrowcolor="#DC2626", font=dict(color="#DC2626", size=11),
        row=1, col=1,
    )

# Net cash flow bars
colours = ["#059669" if v >= 0 else "#DC2626" for v in hist["net"]]
fig.add_trace(go.Bar(
    x=hist["ds"], y=hist["net"],
    name="Net (Historical)",
    marker_color=colours, opacity=0.85,
), row=2, col=1)

fig.add_trace(go.Bar(
    x=fcast["ds"], y=fcast["net"],
    name="Net (Forecast)",
    marker_color=["rgba(37,99,235,.5)" if v >= 0 else "rgba(220,38,38,.5)"
                  for v in fcast["net"]],
    opacity=0.85,
), row=2, col=1)

fig.update_layout(
    height=560,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    legend=dict(orientation="h", y=1.04, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
    font=dict(family="Inter", size=12, color="#334155"),
    hovermode="x unified",
)
fig.update_yaxes(gridcolor="#E2E8F0", gridwidth=1)
fig.update_xaxes(gridcolor="#E2E8F0", gridwidth=1)

st.plotly_chart(fig, use_container_width=True)

# ── Category breakdown ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗂 Spending by Category</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    cat_out = (
        df[df["type"] == "outflow"]
        .groupby("category")["amount"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_pie = px.pie(
        cat_out, names="category", values="amount",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        hole=0.42,
        title="Outflow Categories",
    )
    fig_pie.update_layout(
        paper_bgcolor="#FFFFFF", margin=dict(l=0, r=0, t=40, b=0),
        font=dict(family="Inter", size=12),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    # Monthly trend
    monthly = (
        df.copy()
        .assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "type"])["amount"].sum()
        .reset_index()
    )
    fig_bar = px.bar(
        monthly, x="month", y="amount", color="type",
        barmode="group",
        color_discrete_map={"inflow": "#059669", "outflow": "#DC2626"},
        title="Monthly Inflow vs Outflow",
        labels={"amount": "Amount", "month": "Month", "type": "Type"},
    )
    fig_bar.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(family="Inter", size=12),
        legend=dict(title=""),
    )
    fig_bar.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Financing recommendations ─────────────────────────────────────────────────
st.markdown('<div class="section-header">💡 Financing Recommendations</div>',
            unsafe_allow_html=True)

recs = results["financing_recommendations"]
if recs:
    for r in recs:
        urgency_cls = "urgent" if r["urgency"] == "High" else ""
        badge_cls = "urgent" if r["urgency"] == "High" else ""
        st.markdown(f"""
        <div class="rec-card {urgency_cls}">
          <span class="rec-badge {badge_cls}">{r['urgency']} Priority</span>
          <strong>{r['type']}</strong><br>
          <span style="color:#475569;font-size:.92rem;">{r['reason']}</span><br>
          <span style="color:#64748B;font-size:.85rem;">
            💰 Suggested amount: <strong>{fmt_currency(r['suggested_amount'], currency)}</strong>
            &nbsp;·&nbsp; 📅 Apply by: <strong>{r['apply_by']}</strong>
          </span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("✅ No immediate financing required based on the forecast.")

    st.markdown("""
    <div class="rec-card">
      <span class="rec-badge">Proactive</span>
      <strong>Establish a credit line now</strong><br>
      <span style="color:#475569;font-size:.92rem;">
        Even with healthy cash flow, a revolving credit facility gives you
        a buffer for unexpected seasonal dips without emergency applications.
      </span>
    </div>
    """, unsafe_allow_html=True)

# ── Detailed forecast table ───────────────────────────────────────────────────
with st.expander("📋 Detailed Weekly Forecast Table"):
    display_df = fcast[["ds", "yhat", "yhat_lower", "yhat_upper", "net"]].copy()
    display_df.columns = ["Week", "Projected Balance", "Lower Bound", "Upper Bound", "Net Flow"]
    display_df["Week"] = display_df["Week"].dt.strftime("%Y-%m-%d")
    for col in ["Projected Balance", "Lower Bound", "Upper Bound", "Net Flow"]:
        display_df[col] = display_df[col].apply(
            lambda x: fmt_currency(x, currency)
        )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Download ──────────────────────────────────────────────────────────────────
with st.expander("⬇️ Export Data"):
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_hist = hist.rename(columns={"ds": "week", "balance": "balance",
                                         "net": "net_flow"}).to_csv(index=False)
        st.download_button("Download historical summary (CSV)",
                           csv_hist, "historical_cashflow.csv", "text/csv")

    with col_dl2:
        csv_fcast = fcast[["ds", "yhat", "yhat_lower", "yhat_upper", "net"]].copy()
        csv_fcast.columns = ["week", "projected_balance",
                              "lower_bound", "upper_bound", "net_flow"]
        st.download_button("Download forecast (CSV)",
                           csv_fcast.to_csv(index=False),
                           "cashflow_forecast.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SME Cash Flow AI · Powered by Meta Prophet time-series modelling · "
    "Forecasts are estimates, not financial advice. Consult a qualified advisor."
)

