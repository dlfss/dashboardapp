# app.py
# =============================================================================
# DQ Sales Monitor — Estrutura Completa com Insights Simplificados
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
      .kpi .label {{ font-size: 0.85rem; color: rgba(255,255,255,0.72); margin-bottom: 6px; }}
      .kpi .value {{ font-size: 1.75rem; font-weight: 900; line-height: 1.1; }}
      .kpi .delta {{ font-size: 0.9rem; margin-top: 6px; color: rgba(255,255,255,0.70); }}
      .pill {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 800; }}
      .pill-green {{ background: rgba(0,230,118,0.18); color: {GREEN}; border: 1px solid rgba(0,230,118,0.55); }}
      .pill-red   {{ background: rgba(255,23,68,0.18);  color: {RED};   border: 1px solid rgba(255,23,68,0.55); }}
      .pill-amber {{ background: rgba(255,179,0,0.16);  color: {AMBER}; border: 1px solid rgba(255,179,0,0.45); }}
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 3) MOCK DATA & LOAD
# =============================================================================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]

@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    rows = [
        ("Temperature_is_null", "warn", {"function": "is_not_null"}),
        ("Fuel_Price_is_null", "error", {"function": "is_not_null"}),
        ("Sales_Label_invalid", "error", {"function": "is_in_list"}),
    ]
    return pd.DataFrame(rows, columns=["name", "criticality", "check"])

def _sales_label(v: float, q33: float, q66: float) -> str:
    if v >= q66: return "High"
    if v >= q33: return "Medium"
    return "Low"

@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stores = list(range(1, 46))
    start = datetime(2010, 2, 5)
    dates = _daterange(start, 120)

    rows = []
    id_counter = 1

    for d in dates:
        active_stores = rng.choice(stores, size=rng.integers(25, 46), replace=False)
        for s in active_stores:
            base = rng.normal(1_500_000, 220_000)
            season = 1.0 + 0.10 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi)
            weekly_sales = max(0, base * season + rng.normal(0, 60_000))
            
            rows.append({
                "id": id_counter,
                "Store": int(s),
                "Date": d.date(),
                "Weekly_Sales": float(round(weekly_sales, 2)),
                "Temperature": float(np.clip(rng.normal(55, 18), -10, 90)),
                "Fuel_Price": float(np.clip(rng.normal(3.0, 0.5), 1.5, 5.0)),
                "CPI": float(np.clip(rng.normal(212, 10), 180, 240)),
                "Unemployment": float(np.clip(rng.normal(7.8, 1.5), 4.0, 12.0)),
                "Holiday_Flag": int(rng.random() < 0.08)
            })
            id_counter += 1

    df = pd.DataFrame(rows)
    df_all = df.copy()
    
    # Criar Quarentena Artificial (10%)
    quarantine_idx = rng.choice(df_all.index, size=int(len(df_all) * 0.10), replace=False)
    q = df_all.loc[quarantine_idx].copy()
    v = df_all.drop(quarantine_idx).copy()

    # Adicionar erros dummy
    q["__errors"] = json.dumps({"items": [{"name": "Mock_Error", "message": "Erro simulado"}]})
    q["__warnings"] = None
    v["__errors"] = None
    v["__warnings"] = None

    return v.reset_index(drop=True), q.reset_index(drop=True)

def _normalize_dates_inplace(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def load_data(mode: str):
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()

    if mode == "mock":
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df

    try:
        from pyspark.sql import SparkSession # type: ignore
        spark = SparkSession.getActiveSession()
        if spark is None: raise RuntimeError("SparkSession not found")
        checks = spark.table("databricks_demos.sales_data.dqx_demo_walmart_checks").toPandas()
        valid_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_valid_data").toPandas()
        quarantine_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_quarantine_data").toPandas()
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df
    except Exception:
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df


# =============================================================================
# 4) HELPERS & CHARTS
# =============================================================================
def apply_filters(df: pd.DataFrame, start_date, end_date, selected_stores) -> pd.DataFrame:
    dff = df.copy()
    if "Date" in dff.columns and not np.issubdtype(dff["Date"].dtype, np.datetime64):
        dff["Date"] = pd.to_datetime(dff["Date"], errors="coerce")
    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]
    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]
    return dff

def kpi_card(label: str, value: str, delta: str | None = None, color: str | None = None):
    color_style = f"color: {color};" if color else ""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value" style="{color_style}">{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def base_altair_style(chart):
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="rgba(255,255,255,0.80)",
            titleColor="rgba(255,255,255,0.85)",
            gridColor="rgba(255,255,255,0.08)"
        )
        .configure_legend(labelColor="rgba(255,255,255,0.85)")
    )

def trend_chart(valid_df, quar_df, title):
    data = pd.concat([valid_df.assign(Set="Valid"), quar_df.assign(Set="Quarantine")], ignore_index=True)
    if len(data) == 0: return alt.Chart(pd.DataFrame({"msg": ["Sem dados"]})).mark_text().encode(text="msg:N")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    ts = data.groupby(["Date", "Set"], as_index=False)["Weekly_Sales"].sum().sort_values("Date")
    ts["Set"] = pd.Categorical(ts["Set"], categories=["Valid", "Quarantine"], ordered=True)

    chart = (
        alt.Chart(ts)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("Date:T", title=""),
            y=alt.Y("Weekly_Sales:Q", title="Vendas"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]), legend=alt.Legend(title="", orient="bottom")),
            strokeDash=alt.StrokeDash("Set:N"),
            tooltip=["Date:T", "Set:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
        )
        .properties(height=340, title=title)
    )
    return base_altair_style(chart)

# --- NOVA LÓGICA PARA INSIGHTS SIMPLIFICADOS ---
def get_zone_data(valid_df: pd.DataFrame, quar_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    1. Junta Valid + Quarantine.
    2. Calcula os tercis (33% e 66%) baseados no TOTAL dos dados.
    3. Cria as categorias: Baixa, Média, Alta.
    4. Garante que todas as categorias existem para Valid e Quarantine (para as cores não falharem).
    """
    v = valid_df.copy(); v["Set"] = "Valid"
    q = quar_df.copy(); q["Set"] = "Quarantine"
    full = pd.concat([v, q], ignore_index=True)
    
    full[col] = pd.to_numeric(full[col], errors='coerce')
    full = full.dropna(subset=[col, "Weekly_Sales"])
    
    if len(full) < 5: return pd.DataFrame()

    # Calcular limites
    q33 = full[col].quantile(0.33)
    q66 = full[col].quantile(0.66)

    def classify(val):
        if val <= q33: return "Baixa"
        elif val <= q66: return "Média"
        return "Alta"

    full["Zona"] = full[col].apply(classify)

    # Agrupar
    grouped = full.groupby(["Zona", "Set"], as_index=False)["Weekly_Sales"].mean()

    # Template para garantir que todas as barras aparecem (mesmo que vazias)
    template = pd.DataFrame([
        (z, s) for z in ["Baixa", "Média", "Alta"] for s in ["Valid", "Quarantine"]
    ], columns=["Zona", "Set"])
    
    final = pd.merge(template, grouped, on=["Zona", "Set"], how="left").fillna(0)
    return final

def plot_simple_zone(data: pd.DataFrame, title: str):
    if data.empty:
        return alt.Chart(pd.DataFrame({"msg": ["Sem dados"]})).mark_text().encode(text="msg:N").properties(title=title)
    
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Zona:N", sort=["Baixa", "Média", "Alta"], title=""),
            xOffset=alt.XOffset("Set:N"),
            y=alt.Y("Weekly_Sales:Q", title="Vendas Médias"),
            color=alt.Color("Set:N", 
                            scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]), 
                            legend=alt.Legend(title="", orient="bottom")),
            tooltip=["Zona", "Set", alt.Tooltip("Weekly_Sales", format=",.0f")]
        )
        .properties(height=300, title=title)
    )
    return base_altair_style(chart)


# =============================================================================
# 5) MAIN APP STRUCTURE
# =============================================================================
st.markdown(
    f"""
    # 📊 DQ Sales Monitor
    <span class="pill pill-green">VALID</span>&nbsp;&nbsp;
    <span class="pill pill-red">QUARANTINE</span>&nbsp;&nbsp;
    <span class="pill pill-amber">WARNINGS</span>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ Controlo")
data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"], index=0)
checks_df, valid_df, quarantine_df = load_data(data_mode)

df_all = pd.concat([valid_df.assign(_set="Valid"), quarantine_df.assign(_set="Quarantine")], ignore_index=True)
min_dt, max_dt = df_all["Date"].min(), df_all["Date"].max()
min_date = (min_dt.date() if pd.notna(min_dt) else datetime(2010, 1, 1).date())
max_date = (max_dt.date() if pd.notna(max_dt) else datetime(2010, 12, 31).date())

date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2: start_date, end_date = date_range
else: start_date, end_date = min_date, max_date

store_options = sorted(df_all["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f = apply_filters(quarantine_df, start_date, end_date, selected_stores)

# --- TABS ---
tab_overview, tab_sales, tab_quality, tab_insights, tab_checks = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧠 Insights", "🧱 Checks"]
)

# 1. OVERVIEW
with tab_overview:
    total_valid, total_quar = len(valid_f), len(quar_f)
    total = total_valid + total_quar
    pct_valid = (total_valid / total * 100) if total else 0
    pct_quar = (total_quar / total * 100) if total else 0
    total_sales_valid = valid_f["Weekly_Sales"].sum()
    total_sales_quar = quar_f["Weekly_Sales"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Registos", f"{total:,}".replace(",", "."))
    with c2: kpi_card("Valid", f"{pct_valid:.1f}%", f"{total_valid:,}", color=GREEN)
    with c3: kpi_card("Quarantine", f"{pct_quar:.1f}%", f"{total_quar:,}", color=RED)
    with c4: kpi_card("Vendas (Valid)", f"${total_sales_valid/1e9:.2f}B", color=GREEN)
    with c5: kpi_card("Vendas (Quar.)", f"${total_sales_quar/1e9:.2f}B", color=RED)

    st.markdown("")
    left, right = st.columns([1, 1])
    
    # Donut Chart Simples
    donut_df = pd.DataFrame({"Status": ["Valid", "Quarantine"], "Count": [total_valid, total_quar]})
    donut = alt.Chart(donut_df).mark_arc(innerRadius=75).encode(
        theta="Count:Q",
        color=alt.Color("Status:N", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED])),
        tooltip=["Status", "Count"]
    ).properties(height=340, title="Proporção de Qualidade")
    
    left.altair_chart(base_altair_style(donut), use_container_width=True)
    right.altair_chart(trend_chart(valid_f, quar_f, "Evolução Temporal"), use_container_width=True)

# 2. VENDAS
with tab_sales:
    st.subheader("📈 Vendas")
    if len(valid_f) > 0:
        top = valid_f.groupby("Store", as_index=False)["Weekly_Sales"].sum().sort_values("Weekly_Sales", ascending=False).head(10)
        bar = alt.Chart(top).mark_bar().encode(
            x=alt.X("Weekly_Sales:Q", title="Vendas"),
            y=alt.Y("Store:N", sort="-x", title="Store"),
            color=alt.value(GREEN),
            tooltip=["Store", "Weekly_Sales"]
        ).properties(title="Top Stores (Valid)", height=400)
        st.altair_chart(base_altair_style(bar), use_container_width=True)
    else:
        st.warning("Sem dados Valid.")

# 3. QUALIDADE
with tab_quality:
    st.subheader("✅ Qualidade")
    if len(quar_f) == 0:
        st.success("Tudo limpo!")
    else:
        st.write("Amostra de dados em quarentena:")
        st.dataframe(quar_f.head(50), use_container_width=True)

# 4. INSIGHTS (AQUI ESTÁ A MUDANÇA)
with tab_insights:
    st.subheader("🧠 Insights Simplificados")
    st.caption("Analisa como diferentes fatores afetam as vendas. Tudo dividido em: Baixo, Médio, Alto.")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para apresentar.")
    else:
        # Prepara os dados para cada métrica
        data_temp = get_zone_data(valid_f, quar_f, "Temperature")
        data_fuel = get_zone_data(valid_f, quar_f, "Fuel_Price")
        data_unemp = get_zone_data(valid_f, quar_f, "Unemployment")
        data_cpi = get_zone_data(valid_f, quar_f, "CPI")

        # Linha 1
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(plot_simple_zone(data_temp, "🌡️ Temperatura vs Vendas"), use_container_width=True)
        with c2:
            st.altair_chart(plot_simple_zone(data_fuel, "⛽ Preço Combustível vs Vendas"), use_container_width=True)
        
        # Linha 2
        c3, c4 = st.columns(2)
        with c3:
            st.altair_chart(plot_simple_zone(data_unemp, "💼 Desemprego vs Vendas"), use_container_width=True)
        with c4:
            st.altair_chart(plot_simple_zone(data_cpi, "💰 CPI (Inflação) vs Vendas"), use_container_width=True)

# 5. CHECKS
with tab_checks:
    st.subheader("🧱 Checks")
    st.dataframe(checks_df, use_container_width=True)
