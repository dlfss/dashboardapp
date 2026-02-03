# app.py
# =============================================================================
# DQ Sales Monitor — Visual “3 segundos”
# =============================================================================

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# =============================================================================
# 1) CONFIG GLOBAL
# =============================================================================
st.set_page_config(page_title="DQ Sales Monitor", page_icon="📊", layout="wide")

GREEN = "#00E676"
RED = "#FF1744"
AMBER = "#FFB300"

try:
    alt.themes.enable("dark")
except Exception:
    pass


# =============================================================================
# 2) CSS / UI
# =============================================================================
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; }}
      h1, h2, h3 {{ letter-spacing: -0.02em; }}

      .kpi {{
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
      }}
      .kpi .label {{
        font-size: 0.85rem;
        color: rgba(255,255,255,0.72);
        margin-bottom: 6px;
      }}
      .kpi .value {{
        font-size: 1.75rem;
        font-weight: 900;
        line-height: 1.1;
      }}
      .kpi .delta {{
        font-size: 0.9rem;
        margin-top: 6px;
        color: rgba(255,255,255,0.70);
      }}

      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.01em;
      }}
      .pill-green {{ background: rgba(0,230,118,0.18); color: {GREEN}; border: 1px solid rgba(0,230,118,0.55); }}
      .pill-red   {{ background: rgba(255,23,68,0.18);  color: {RED};   border: 1px solid rgba(255,23,68,0.55); }}
      .pill-amber {{ background: rgba(255,179,0,0.16);  color: {AMBER}; border: 1px solid rgba(255,179,0,0.45); }}
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 3) MOCK DATA
# =============================================================================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]

@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    rows = [
        ("Sales_Label_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Sales_Label"}}),
        ("Fuel_Price_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Fuel_Price"}}),
        ("Temperature_isnt_in_range", "error", {"function": "is_in_range", "arguments": {"column": "Temperature", "min_limit": -10, "max_limit": 55}}),
        ("Store_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Store"}}),
        ("CPI_is_null", "error", {"function": "is_not_null", "arguments": {"column": "CPI"}}),
    ]
    df = pd.DataFrame(rows, columns=["name", "criticality", "check"])
    df["check"] = df["check"].apply(lambda x: json.dumps(x))
    return df

@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stores = list(range(1, 46))
    dates = _daterange(datetime(2010, 2, 5), 120)
    rows = []
    
    for d in dates:
        active_stores = rng.choice(stores, size=rng.integers(25, 46), replace=False)
        for s in active_stores:
            rows.append({
                "Store": int(s), "Date": d.date(),
                "Weekly_Sales": max(0, rng.normal(1_500_000, 220_000)),
                "Holiday_Flag": int(rng.random() < 0.08),
                "Temperature": float(round(rng.normal(55, 18), 2)),
                "Fuel_Price": float(round(rng.normal(2.75, 0.35), 3)),
                "CPI": float(round(rng.normal(212, 4.0), 6)),
                "Unemployment": float(round(rng.normal(7.8, 0.6), 3)),
            })
    df = pd.DataFrame(rows)
    quarantine_idx = rng.choice(df.index, size=int(len(df) * 0.10), replace=False)
    q = df.loc[quarantine_idx].copy()
    v = df.drop(quarantine_idx).copy()
    
    q["__errors"] = json.dumps({"items": [{"name": "Generic_Error", "column": "Multiple", "message": "DQ Issue"}]})
    q["__warnings"] = json.dumps({"items": []})
    v["__errors"], v["__warnings"] = None, None
    return v, q


# =============================================================================
# 4) LOAD & HELPERS
# =============================================================================
def load_data(mode: str):
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()
    for df in [valid_df, quarantine_df]:
        df["Date"] = pd.to_datetime(df["Date"])
    return checks, valid_df, quarantine_df

def apply_filters(df, start_date, end_date, selected_stores):
    dff = df.copy()
    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]
    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]
    return dff

def kpi_card(label, value, delta=None, color=None):
    color_style = f"color: {color};" if color else ""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value" style="{color_style}">{value}</div>{delta_html}</div>', unsafe_allow_html=True)

def base_altair_style(chart):
    return chart.configure_view(strokeOpacity=0).configure_axis(labelColor="rgba(255,255,255,0.80)", titleColor="rgba(255,255,255,0.85)", gridColor="rgba(255,255,255,0.08)").configure_legend(labelColor="rgba(255,255,255,0.85)")


# =============================================================================
# 5) MAIN APP & SIDEBAR
# =============================================================================
st.markdown(f'# 📊 DQ Sales Monitor <br><span class="pill pill-green">VALID</span> <span class="pill pill-red">QUARANTINE</span>', unsafe_allow_html=True)

data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"])
checks_df, valid_df, quarantine_df = load_data(data_mode)

df_all = pd.concat([valid_df, quarantine_df])
date_range = st.sidebar.date_input("Datas", value=(df_all["Date"].min().date(), df_all["Date"].max().date()))
selected_stores = st.sidebar.multiselect("Store", sorted(df_all["Store"].unique()))

valid_f = apply_filters(valid_df, date_range[0], date_range[1], selected_stores)
quar_f = apply_filters(quarantine_df, date_range[0], date_range[1], selected_stores)

tab_overview, tab_sales, tab_quality, tab_insights, tab_checks = st.tabs(["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧠 Insights", "🧱 Checks"])


# =============================================================================
# 6) TABS CONTENT (Insights Alterado conforme pedido)
# =============================================================================

with tab_overview:
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Total Registos", f"{len(valid_f)+len(quar_f):,}")
    with c2: kpi_card("Valid", f"{(len(valid_f)/(len(valid_f)+len(quar_f))*100):.1f}%", color=GREEN)
    with c3: kpi_card("Quarantine", f"{(len(quar_f)/(len(valid_f)+len(quar_f))*100):.1f}%", color=RED)

with tab_insights:
    st.subheader("🧠 Insights Dinâmicos")
    st.caption("Análise de correlação com média móvel e visualização em abas.")

    if valid_f.empty and quar_f.empty:
        st.warning("Sem dados.")
    else:
        # Layout inspirado na imagem enviada
        with st.container(border=True):
            col_sel, col_tog = st.columns([2, 1])
            with col_sel:
                metrics = ["Temperature", "Fuel_Price", "Unemployment", "CPI"]
                selected_metric = st.selectbox("Selecione a Métrica", metrics)
            with col_tog:
                st.write("")
                use_rolling = st.toggle("Média Móvel (7 dias)", value=True)

        # Preparação de dados
        v_line = valid_f.groupby("Date")[[selected_metric, "Weekly_Sales"]].mean().reset_index().assign(Set="Valid")
        q_line = quar_f.groupby("Date")[[selected_metric, "Weekly_Sales"]].mean().reset_index().assign(Set="Quarantine")
        combined = pd.concat([v_line, q_line]).sort_values("Date")

        if use_rolling:
            combined["Weekly_Sales"] = combined.groupby("Set")["Weekly_Sales"].transform(lambda x: x.rolling(7).mean())

        t_chart, t_data = st.tabs(["📊 Gráfico", "📑 Dados"])
        
        with t_chart:
            chart = alt.Chart(combined).mark_line(strokeWidth=3).encode(
                x="Date:T", y="Weekly_Sales:Q", color=alt.Color("Set:N", scale=alt.Scale(range=[GREEN, RED])),
                tooltip=["Date", "Weekly_Sales"]
            ).properties(height=400)
            st.altair_chart(base_altair_style(chart), use_container_width=True)
            
        with t_data:
            st.dataframe(combined, use_container_width=True)

with tab_sales: st.write("Conteúdo de Vendas...")
with tab_quality: st.write("Conteúdo de Qualidade...")
with tab_checks: st.dataframe(checks_df, use_container_width=True)
