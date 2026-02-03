# app.py
# DQ Sales Monitor — Visual "3 segundos"
# Ideia: dashboard rápido p/ decidir + separar “valid” vs “quarantine”

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# =========================
# 1) CONFIG GLOBAL
# =========================
st.set_page_config(page_title="DQ Sales Monitor", page_icon="📊", layout="wide")
# -> layout wide porque este tipo de dashboard morre se tiver scroll horizontal

GREEN = "#00E676"   # -> “ok / limpo” (cor forte = leitura rápida)
RED   = "#FF1744"   # -> “problema / quarentena”
AMBER = "#FFB300"   # -> “warning / atenção”

try:
    alt.themes.enable("dark")
    # -> tenta tema dark do Altair (se existir na versão). Não é obrigatório.
except Exception:
    pass


# =========================
# 2) CSS / UI
# =========================
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; }}
      /* -> só melhora o “respirar” do layout */

      .kpi {{
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
      }}
      /* -> KPI card “premium”: dá logo cara de produto */

      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 800;
      }}
      /* -> pills = legenda rápida sem texto extra */

      .pill-green {{ background: rgba(0,230,118,0.18); color: {GREEN}; border: 1px solid rgba(0,230,118,0.55); }}
      .pill-red   {{ background: rgba(255,23,68,0.18);  color: {RED};   border: 1px solid rgba(255,23,68,0.55); }}
      .pill-amber {{ background: rgba(255,179,0,0.16);  color: {AMBER}; border: 1px solid rgba(255,179,0,0.45); }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 3) MOCK DATA
# =========================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]
    # -> datas semanais para poderes fazer trend “tipo Walmart dataset”


@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    rows = [
        ("Sales_Label_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Sales_Label"}}),
        ("Sales_Label_other_value", "error", {"function": "is_in_list", "arguments": {"column": "Sales_Label", "allowed": ["High", "Medium", "Low"]}}),
        ("Holiday_Flag_is_null", "warn", {"function": "is_not_null", "arguments": {"column": "Holiday_Flag"}}),
        # ... (o resto são regras típicas de DQ)
    ]
    df = pd.DataFrame(rows, columns=["name", "criticality", "check"])

    df["filter"] = None
    df["run_config_name"] = "default"
    # -> metadata “fingida” mas realista (para parecer catálogo de checks)

    df["check"] = df["check"].apply(lambda x: json.dumps(x))
    # -> simula o formato real: guardas JSON como string (como viria de uma tabela)
    return df


def _sales_label(v: float, q33: float, q66: float) -> str:
    if v >= q66:
        return "High"
    if v >= q33:
        return "Medium"
    return "Low"
    # -> cria uma label categórica com tercis (e depois dá para criar erros “label inválida”)


@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # -> random generator “moderno” e reprodutível

    stores = list(range(1, 46))
    # -> 45 stores, lembra dataset Walmart (boa referência mental)

    start = datetime(2010, 2, 5)
    weeks = 120
    dates = _daterange(start, weeks)

    rows = []
    id_counter = 1

    for d in dates:
        active_stores = rng.choice(stores, size=rng.integers(25, 46), replace=False)
        # -> nem todas as lojas em todas as semanas (fica mais real)

        for s in active_stores:
            base = rng.normal(1_500_000, 220_000)
            season = 1.0 + 0.10 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi)
            # -> sazonalidade simples, só para não ser “linha reta”

            holiday = int(rng.random() < 0.08)
            holiday_boost = 1.0 + (0.12 if holiday else 0.0)
            # -> feriados dão boost de vendas (fica intuitivo nos gráficos)

            weekly_sales = max(0, base * season * holiday_boost + rng.normal(0, 60_000))
            # -> garante que não tens vendas negativas (mock “saudável”)

            temperature  = float(np.clip(rng.normal(55, 18), -10, 55))
            fuel_price   = float(np.clip(rng.normal(2.75, 0.35), 1.5, 4.5))
            cpi          = float(np.clip(rng.normal(212, 4.0), 200, 230))
            unemployment = float(np.clip(rng.normal(7.8, 0.6), 5.5, 10.5))
            # -> “features” plausíveis para powers insights (e também criar checks)

            rows.append(
                {
                    "id": id_counter,
                    "Store": int(s),
                    "Date": d.date(),
                    "Weekly_Sales": float(round(weekly_sales, 2)),
                    "Holiday_Flag": holiday,
                    "Temperature": float(round(temperature, 2)),
                    "Fuel_Price": float(round(fuel_price, 3)),
                    "CPI": float(round(cpi, 6)),
                    "Unemployment": float(round(unemployment, 3)),
                }
            )
            id_counter += 1

    df = pd.DataFrame(rows)

    q33, q66 = df["Weekly_Sales"].quantile([0.33, 0.66]).tolist()
    df["Sales_Label"] = df["Weekly_Sales"].apply(lambda v: _sales_label(v, q33, q66))
    # -> cria labels consistentes com a distribuição real do mock

    quarantine_idx = rng.choice(df.index, size=int(len(df) * 0.10), replace=False)
    # -> 10% vai para “quarantine” (simula pipeline DQ)
    q = df.loc[quarantine_idx].copy()
    v = df.drop(quarantine_idx).copy()

    def add_issue(row: pd.Series) -> tuple[dict, dict]:
        errors, warnings = [], []

        issue_type = rng.choice(
            ["temp_null", "temp_range", "fuel_null", "holiday_invalid", "label_null", "label_invalid"],
            p=[0.18, 0.34, 0.20, 0.10, 0.08, 0.10],
        )
        # -> controlas “mix” de problemas para os gráficos não ficarem vazios

        def err(name, col, msg):
            errors.append({"name": name, "column": col, "message": msg})

        def warn(name, col, msg):
            warnings.append({"name": name, "column": col, "message": msg})

        if issue_type == "temp_null":
            row["Temperature"] = None
            warn("Temperature_is_null", "Temperature", "Temperature vazio.")
        elif issue_type == "temp_range":
            row["Temperature"] = float(rng.choice([-25, 80, 120]))
            err("Temperature_isnt_in_range", "Temperature", "Temperature fora do intervalo [-10, 55].")
        elif issue_type == "fuel_null":
            row["Fuel_Price"] = None
            err("Fuel_Price_is_null", "Fuel_Price", "Fuel_Price vazio.")
        elif issue_type == "holiday_invalid":
            row["Holiday_Flag"] = int(rng.choice([2, 3, -1]))
            err("Holiday_Flag_other_value", "Holiday_Flag", "Holiday_Flag fora de {0,1}.")
        elif issue_type == "label_null":
            row["Sales_Label"] = None
            err("Sales_Label_is_null", "Sales_Label", "Sales_Label vazio.")
        elif issue_type == "label_invalid":
            row["Sales_Label"] = rng.choice(["HIGH", "Med", "Unknown"])
            err("Sales_Label_other_value", "Sales_Label", "Sales_Label inválido.")

        return {"items": errors}, {"items": warnings}

    q_errors, q_warnings, q2 = [], [], []
    for _, r in q.iterrows():
        r = r.copy()
        e, w = add_issue(r)
        q_errors.append(json.dumps(e))
        q_warnings.append(json.dumps(w))
        # -> guardas logs como JSON string, depois o parse_issue_counts conta as regras
        q2.append(r)

    q = pd.DataFrame(q2)
    q["__errors"] = q_errors
    q["__warnings"] = q_warnings

    v["__errors"] = None
    v["__warnings"] = None
    # -> valid não tem logs (fica limpinho para decisões)

    return v.reset_index(drop=True), q.reset_index(drop=True)


# =========================
# 4) LOAD (Mock vs Databricks)
# =========================
def _normalize_dates_inplace(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # -> deixa Date pronto para filtros e charts (e evita conversões repetidas)
    return df


def load_data(mode: str):
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()
    # -> começa sempre com mock, assim nunca ficas “sem nada” se Spark falhar

    if mode == "mock":
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df

    try:
        from pyspark.sql import SparkSession  # type: ignore
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("SparkSession not found")

        checks = spark.table("databricks_demos.sales_data.dqx_demo_walmart_checks").toPandas()
        valid_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_valid_data").toPandas()
        quarantine_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_quarantine_data").toPandas()
        # -> se estás mesmo em Databricks, isto lê as tabelas reais

        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df

    except Exception:
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df
        # -> fallback silencioso (ux boa: app nunca “morre”)


# =========================
# 5) HELPERS
# =========================
def apply_filters(df: pd.DataFrame, start_date, end_date, selected_stores) -> pd.DataFrame:
    dff = df.copy()

    if "Date" in dff.columns and not np.issubdtype(dff["Date"].dtype, np.datetime64):
        dff["Date"] = pd.to_datetime(dff["Date"], errors="coerce")
        # -> segurança: se vier string, converte aqui

    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]
    # -> filtro inclusive (start e end entram)

    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]
        # -> só filtra stores se o user escolheu alguma (senão deixas tudo)
    return dff


def parse_issue_counts(series: pd.Series) -> pd.DataFrame:
    counts = {}
    for x in series.dropna():
        try:
            payload = json.loads(x)
            for item in payload.get("items", []):
                name = item.get("name", "unknown")
                counts[name] = counts.get(name, 0) + 1
                # -> aqui é o coração do “quais regras estão a disparar”
        except Exception:
            continue

    df = pd.DataFrame({"Regra": list(counts.keys()), "Ocorrências": list(counts.values())})
    if len(df):
        df = df.sort_values("Ocorrências", ascending=False).reset_index(drop=True)
        # -> ordena para o gráfico mostrar o que importa primeiro
    return df


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
    # -> KPI “grande” e legível em 1 segundo, muito bom para o Overview


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
    # -> garante consistência: todos os charts ficam com o mesmo “look”


# =========================
# 8) HEADER
# =========================
st.markdown(
    f"""
    # 📊 DQ Sales Monitor
    <span class="pill pill-green">VALID</span>&nbsp;&nbsp;
    <span class="pill pill-red">QUARANTINE</span>&nbsp;&nbsp;
    <span class="pill pill-amber">WARNINGS</span>
    """,
    unsafe_allow_html=True
)
# -> logo no topo já ensinas a legenda com cor (sem explicar em texto)


# =========================
# 9) SIDEBAR
# =========================
st.sidebar.header("⚙️ Controlo")

data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"], index=0)
# -> o user escolhe a fonte sem complicar

checks_df, valid_df, quarantine_df = load_data(data_mode)
# -> carrega tudo e garante que tens fallback

df_all = pd.concat([valid_df.assign(_set="Valid"), quarantine_df.assign(_set="Quarantine")], ignore_index=True)
# -> df_all serve para descobrir datas e stores disponíveis (sem duplicar lógica)

min_dt = df_all["Date"].min()
max_dt = df_all["Date"].max()
min_date = (min_dt.date() if pd.notna(min_dt) else datetime(2010, 1, 1).date())
max_date = (max_dt.date() if pd.notna(max_dt) else datetime(2010, 12, 31).date())
# -> proteção caso Date esteja vazio/NaT

date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date
# -> fallback defensivo (Streamlit às vezes devolve um valor só)

store_options = sorted(df_all["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f  = apply_filters(quarantine_df, start_date, end_date, selected_stores)
# -> filtros aplicados aos dois sets, assim o “comparar” fica justo


# =========================
# 10) TABS
# =========================
tab_overview, tab_sales, tab_quality, tab_insights, tab_checks = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧠 Insights", "🧱 Checks (Advanced)"]
)
# -> tab = separa públicos: executivo/analista/auditor


# =========================
# 11) OVERVIEW
# =========================
with tab_overview:
    total_valid = len(valid_f)
    total_quar  = len(quar_f)
    total = total_valid + total_quar

    pct_valid = (total_valid / total * 100) if total else 0
    pct_quar  = (total_quar  / total * 100) if total else 0
    # -> percentagens para “ver a saúde” num relance

    total_sales_valid = float(pd.to_numeric(valid_f.get("Weekly_Sales", pd.Series(dtype=float)), errors="coerce").sum()) if total_valid else 0.0
    total_sales_quar  = float(pd.to_numeric(quar_f.get("Weekly_Sales", pd.Series(dtype=float)), errors="coerce").sum()) if total_quar else 0.0
    # -> soma de vendas por set (to_numeric evita crash se vier string)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Registos", f"{total:,}".replace(",", "."))
    with c2:
        kpi_card("Valid", f"{pct_valid:.1f}%", f"{total_valid:,} reg.".replace(",", "."), color=GREEN)
    with c3:
        kpi_card("Quarantine", f"{pct_quar:.1f}%", f"{total_quar:,} reg.".replace(",", "."), color=RED)
    with c4:
        kpi_card("Vendas (Valid)", f"${total_sales_valid/1e9:.2f}B", color=GREEN)
    with c5:
        kpi_card("Vendas (Quarantine)", f"${total_sales_quar/1e9:.2f}B", color=RED)
    # -> esta linha de KPIs é literalmente o “3 segundos”

    # ... (aqui entram donut/trend/offenders, que seguem a mesma lógica: comparação clara)
