# app.py
# =============================================================================
# DQ Sales Monitor — Visual “3 segundos”
# =============================================================================
# Objetivo:
# - Comparar rapidamente dados "Valid" vs "Quarantine"
# - KPIs + trend + top offenders (visão executiva)
# - Drilldown: Vendas / Qualidade / Insights / Checks
#
# Nota importante (Altair v6 / Streamlit Cloud):
# - Evitamos alt.Chart(...).transform_bin(...) porque em Altair v6 pode dar
#   SchemaValidationError dependendo da assinatura/schema.
# - Fazemos o binning em pandas (robusto em qualquer versão) e só plotamos.

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

GREEN = "#00E676"   # Valid
RED = "#FF1744"     # Quarantine
AMBER = "#FFB300"   # Warnings

# Tema dark do Altair pode variar por versão; se falhar, seguimos.
try:
    alt.themes.enable("dark")
except Exception:
    pass


# =============================================================================
# 2) CSS / UI (cards KPI + pills)
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
# 3) MOCK DATA (fallback)
# =============================================================================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    """Lista de datas semanais."""
    return [start + timedelta(days=7 * i) for i in range(weeks)]


@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    """
    Simula catálogo de checks DQ.
    - 'check' é JSON string para imitar o formato real.
    """
    rows = [
        ("Sales_Label_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Sales_Label"}}),
        ("Sales_Label_other_value", "error", {"function": "is_in_list", "arguments": {"column": "Sales_Label", "allowed": ["High", "Medium", "Low"]}}),
        ("Holiday_Flag_is_null", "warn", {"function": "is_not_null", "arguments": {"column": "Holiday_Flag"}}),
        ("Fuel_Price_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Fuel_Price"}}),
        ("Temperature_isnt_in_range", "error", {"function": "is_in_range", "arguments": {"column": "Temperature", "min_limit": -10, "max_limit": 55}}),
        ("Temperature_is_null", "warn", {"function": "is_not_null", "arguments": {"column": "Temperature"}}),
        ("Unemployment_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Unemployment"}}),
        ("id_is_null", "error", {"function": "is_not_null", "arguments": {"column": "id"}}),
        ("id_isnt_in_range", "error", {"function": "is_in_range", "arguments": {"column": "id", "min_limit": 1, "max_limit": 6435}}),
        ("Store_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Store"}}),
        ("Store_isnt_in_range", "error", {"function": "is_in_range", "arguments": {"column": "Store", "min_limit": 1, "max_limit": 45}}),
        ("Date_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Date"}}),
        ("Weekly_Sales_is_null", "error", {"function": "is_not_null", "arguments": {"column": "Weekly_Sales"}}),
        ("Holiday_Flag_other_value", "error", {"function": "is_in_list", "arguments": {"column": "Holiday_Flag", "allowed": [0, 1]}}),
        ("CPI_is_null", "error", {"function": "is_not_null", "arguments": {"column": "CPI"}}),
    ]
    df = pd.DataFrame(rows, columns=["name", "criticality", "check"])
    df["filter"] = None
    df["run_config_name"] = "default"
    df["check"] = df["check"].apply(lambda x: json.dumps(x))
    return df


def _sales_label(v: float, q33: float, q66: float) -> str:
    """High/Medium/Low por tercis das vendas (só para mock)."""
    if v >= q66:
        return "High"
    if v >= q33:
        return "Medium"
    return "Low"


@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera dados realistas e separa em:
    - valid (~90%)
    - quarantine (~10%) com issues e logs JSON (__errors/__warnings)
    """
    rng = np.random.default_rng(seed)
    stores = list(range(1, 46))
    start = datetime(2010, 2, 5)
    weeks = 120
    dates = _daterange(start, weeks)

    rows = []
    id_counter = 1

    for d in dates:
        active_stores = rng.choice(stores, size=rng.integers(25, 46), replace=False)

        for s in active_stores:
            base = rng.normal(1_500_000, 220_000)
            season = 1.0 + 0.10 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi)

            holiday = int(rng.random() < 0.08)
            holiday_boost = 1.0 + (0.12 if holiday else 0.0)

            weekly_sales = max(0, base * season * holiday_boost + rng.normal(0, 60_000))

            # Features plausíveis (clip para valid)
            temperature = float(np.clip(rng.normal(55, 18), -10, 55))
            fuel_price = float(np.clip(rng.normal(2.75, 0.35), 1.5, 4.5))
            cpi = float(np.clip(rng.normal(212, 4.0), 200, 230))
            unemployment = float(np.clip(rng.normal(7.8, 0.6), 5.5, 10.5))

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

    # Só para mock: etiqueta High/Medium/Low de vendas
    q33, q66 = df["Weekly_Sales"].quantile([0.33, 0.66]).tolist()
    df["Sales_Label"] = df["Weekly_Sales"].apply(lambda v: _sales_label(v, q33, q66))

    # 10% em quarentena
    quarantine_idx = rng.choice(df.index, size=int(len(df) * 0.10), replace=False)
    q = df.loc[quarantine_idx].copy()
    v = df.drop(quarantine_idx).copy()

    def add_issue(row: pd.Series) -> tuple[dict, dict]:
        """Injeta um issue por linha e devolve logs errors/warnings."""
        errors, warnings = [], []

        issue_type = rng.choice(
            ["temp_null", "temp_range", "fuel_null", "holiday_invalid", "label_null", "label_invalid"],
            p=[0.18, 0.34, 0.20, 0.10, 0.08, 0.10],
        )

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
        q2.append(r)

    q = pd.DataFrame(q2)
    q["__errors"] = q_errors
    q["__warnings"] = q_warnings

    v["__errors"] = None
    v["__warnings"] = None

    return v.reset_index(drop=True), q.reset_index(drop=True)


# =============================================================================
# 4) LOAD (Mock vs Databricks)
# =============================================================================
def _normalize_dates_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Date -> datetime (para filtros e charts)."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_data(mode: str):
    """
    Carrega checks + valid + quarantine.
    - Começa sempre com mock
    - Se mode==databricks: tenta Spark / tabelas reais
    - Se falhar: devolve mock (sem crash)
    """
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()

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

        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df

    except Exception:
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df


# =============================================================================
# 5) HELPERS (filtros + parsing + KPI + estilo Altair)
# =============================================================================
def apply_filters(df: pd.DataFrame, start_date, end_date, selected_stores) -> pd.DataFrame:
    """
    Filtra por:
    - intervalo de datas (inclusive)
    - stores selecionadas (opcional)
    """
    dff = df.copy()

    if "Date" in dff.columns and not np.issubdtype(dff["Date"].dtype, np.datetime64):
        dff["Date"] = pd.to_datetime(dff["Date"], errors="coerce")

    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]

    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]

    return dff


def parse_issue_counts(series: pd.Series) -> pd.DataFrame:
    """Conta ocorrências de rules a partir de JSON string em __errors/__warnings."""
    counts = {}

    for x in series.dropna():
        try:
            payload = json.loads(x)
            for item in payload.get("items", []):
                name = item.get("name", "unknown")
                counts[name] = counts.get(name, 0) + 1
        except Exception:
            continue

    df = pd.DataFrame({"Regra": list(counts.keys()), "Ocorrências": list(counts.values())})
    if len(df):
        df = df.sort_values("Ocorrências", ascending=False).reset_index(drop=True)
    return df


def kpi_card(label: str, value: str, delta: str | None = None, color: str | None = None):
    """Card KPI em HTML para ficar “produto” e não “demo”. """
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
    """Estilo base: eixos/legendas para UI escura e look consistente."""
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="rgba(255,255,255,0.80)",
            titleColor="rgba(255,255,255,0.85)",
            gridColor="rgba(255,255,255,0.08)"
        )
        .configure_legend(labelColor="rgba(255,255,255,0.85)")
    )


# =============================================================================
# 6) INSIGHTS (robustos): binning em pandas (sem Altair transform_bin)
# =============================================================================
def _prep_insight_df(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Junta Valid+Quarantine num DF e garante tipos numéricos.
    Remove NaNs (para não estragar charts).
    """
    data = pd.concat([valid_df.assign(Set="Valid"), quar_df.assign(Set="Quarantine")], ignore_index=True)

    if feature not in data.columns or "Weekly_Sales" not in data.columns:
        return pd.DataFrame(columns=[feature, "Weekly_Sales", "Set", "Date", "Store"])

    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data["Weekly_Sales"] = pd.to_numeric(data["Weekly_Sales"], errors="coerce")

    keep_cols = [c for c in [feature, "Weekly_Sales", "Set", "Date", "Store"] if c in data.columns]
    data = data[keep_cols].dropna(subset=[feature, "Weekly_Sales", "Set"])

    data["Set"] = pd.Categorical(data["Set"], categories=["Valid", "Quarantine"], ordered=True)
    return data


def _binned_effect_df(data: pd.DataFrame, feature: str, maxbins: int = 18, agg: str = "mean") -> pd.DataFrame:
    """
    Faz bins em pandas (compatível com qualquer Altair).
    Output: Set | bin_left | bin_right | xmid | sales | n
    """
    d = data.copy()

    # Segurança: se houver tipos esquisitos, converte
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d["Weekly_Sales"] = pd.to_numeric(d["Weekly_Sales"], errors="coerce")
    d = d.dropna(subset=[feature, "Weekly_Sales", "Set"])

    if len(d) < 10:
        return pd.DataFrame(columns=["Set", "bin_left", "bin_right", "xmid", "sales", "n"])

    x = d[feature].to_numpy()
    edges = np.histogram_bin_edges(x, bins=maxbins)

    # Se não houver variação suficiente, não dá para binar
    if len(edges) < 3 or np.allclose(edges[0], edges[-1]):
        return pd.DataFrame(columns=["Set", "bin_left", "bin_right", "xmid", "sales", "n"])

    d["bin"] = pd.cut(d[feature], bins=edges, include_lowest=True)

    g = (
        d.groupby(["Set", "bin"], as_index=False)
        .agg(
            n=("Weekly_Sales", "size"),
            mean_sales=("Weekly_Sales", "mean"),
            median_sales=("Weekly_Sales", "median"),
        )
    )

    g["bin_left"] = g["bin"].apply(lambda iv: float(iv.left) if pd.notna(iv) else np.nan)
    g["bin_right"] = g["bin"].apply(lambda iv: float(iv.right) if pd.notna(iv) else np.nan)
    g["xmid"] = (g["bin_left"] + g["bin_right"]) / 2.0
    g["sales"] = g["mean_sales"] if agg == "mean" else g["median_sales"]

    g = g.sort_values(["Set", "xmid"]).reset_index(drop=True)
    return g[["Set", "bin_left", "bin_right", "xmid", "sales", "n"]]


def feature_effect_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str, title: str,
                         maxbins: int = 18, agg: str = "mean"):
    """
    Curva de impacto:
    - binning em pandas
    - linha com pontos (Valid vs Quarantine)
    """
    data = _prep_insight_df(valid_df, quar_df, feature)
    b = _binned_effect_df(data, feature=feature, maxbins=maxbins, agg=agg)

    if len(b) == 0:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados suficientes"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=320, title=title)
        )

    chart = (
        alt.Chart(b)
        .mark_line(point=True, strokeWidth=4)
        .encode(
            x=alt.X("xmid:Q", title=feature),
            y=alt.Y("sales:Q", title=f"Weekly_Sales ({agg})"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=[
                "Set:N",
                alt.Tooltip("n:Q", title="N", format=","),
                alt.Tooltip("sales:Q", title=f"Weekly_Sales ({agg})", format=",.0f"),
                alt.Tooltip("bin_left:Q", title="bin left", format=".2f"),
                alt.Tooltip("bin_right:Q", title="bin right", format=".2f"),
            ],
        )
        .properties(height=320, title=title)
    )
    return base_altair_style(chart)


def feature_distribution_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str, title: str, maxbins: int = 30):
    """
    Histograma do feature (pandas bins iguais para ambos sets):
    - mostra onde Quarantine “concentra”
    """
    data = _prep_insight_df(valid_df, quar_df, feature)
    if len(data) < 10:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados suficientes"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=240, title=title)
        )

    x = data[feature].to_numpy()
    edges = np.histogram_bin_edges(x, bins=maxbins)
    if len(edges) < 3 or np.allclose(edges[0], edges[-1]):
        return (
            alt.Chart(pd.DataFrame({"msg": ["Feature sem variação suficiente"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=240, title=title)
        )

    d = data.copy()
    d["bin"] = pd.cut(d[feature], bins=edges, include_lowest=True)

    g = (
        d.groupby(["Set", "bin"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    g["bin_left"] = g["bin"].apply(lambda iv: float(iv.left) if pd.notna(iv) else np.nan)
    g["bin_right"] = g["bin"].apply(lambda iv: float(iv.right) if pd.notna(iv) else np.nan)
    g["xmid"] = (g["bin_left"] + g["bin_right"]) / 2.0

    chart = (
        alt.Chart(g)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("xmid:Q", title=feature),
            y=alt.Y("count:Q", title="Registos"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Set:N", alt.Tooltip("count:Q", format=","), "bin_left:Q", "bin_right:Q"],
        )
        .properties(height=240, title=title)
    )
    return base_altair_style(chart)


def feature_scatter_trend_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str, title: str, sample_max: int = 6000):
    """
    Scatter + LOESS:
    - bom para validar padrão “real”
    - com amostragem para performance
    """
    data = _prep_insight_df(valid_df, quar_df, feature)
    if len(data) < 30:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados suficientes"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=320, title=title)
        )

    if len(data) > sample_max:
        data = data.sample(sample_max, random_state=42)

    points = (
        alt.Chart(data)
        .mark_circle(size=28, opacity=0.25)
        .encode(
            x=alt.X(f"{feature}:Q", title=feature),
            y=alt.Y("Weekly_Sales:Q", title="Weekly_Sales"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Set:N", alt.Tooltip(f"{feature}:Q"), alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
        )
    )

    trend = (
        alt.Chart(data)
        .transform_loess(feature, "Weekly_Sales", groupby=["Set"])
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X(f"{feature}:Q"),
            y=alt.Y("Weekly_Sales:Q"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=None,
            ),
        )
    )

    return base_altair_style((points + trend).properties(height=320, title=title))


# =============================================================================
# 7) OUTROS CHARTS (holiday + trend + offenders)
# =============================================================================
def grouped_holiday_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, title: str):
    """Comparação Sem feriado vs Com feriado (média) para Valid vs Quarantine."""
    def holiday_avg(df: pd.DataFrame, set_label: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=["Holiday", "AvgSales", "Set"])

        d = df.copy()
        d["Holiday_Flag"] = pd.to_numeric(d.get("Holiday_Flag"), errors="coerce")
        d = d[d["Holiday_Flag"].isin([0, 1])]
        if len(d) == 0:
            return pd.DataFrame(columns=["Holiday", "AvgSales", "Set"])

        g = d.groupby("Holiday_Flag", as_index=False)["Weekly_Sales"].mean()
        g["Holiday"] = g["Holiday_Flag"].map({0: "Sem feriado", 1: "Com feriado"})
        g = g.rename(columns={"Weekly_Sales": "AvgSales"})
        g["Set"] = set_label
        return g[["Holiday", "AvgSales", "Set"]]

    data = pd.concat([holiday_avg(valid_df, "Valid"), holiday_avg(quar_df, "Quarantine")], ignore_index=True)

    if len(data) == 0:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=360, title=title)
        )

    data["Set"] = pd.Categorical(data["Set"], categories=["Valid", "Quarantine"], ordered=True)

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Holiday:N", title="", sort=["Sem feriado", "Com feriado"]),
            xOffset=alt.XOffset("Set:N"),
            y=alt.Y("AvgSales:Q", title="Média Weekly Sales"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Set:N", "Holiday:N", alt.Tooltip("AvgSales:Q", format=",.0f")],
        )
        .properties(height=360, title=title)
    )
    return base_altair_style(chart)


def trend_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, title: str):
    """Trend temporal (soma por semana) para comparação rápida Valid vs Quarantine."""
    data = pd.concat([valid_df.assign(Set="Valid"), quar_df.assign(Set="Quarantine")], ignore_index=True)

    if len(data) == 0:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=340, title=title)
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    ts = data.groupby(["Date", "Set"], as_index=False)["Weekly_Sales"].sum().sort_values("Date")
    ts["Set"] = pd.Categorical(ts["Set"], categories=["Valid", "Quarantine"], ordered=True)

    chart = (
        alt.Chart(ts)
        .mark_line(strokeWidth=5)
        .encode(
            x=alt.X("Date:T", title=""),
            y=alt.Y("Weekly_Sales:Q", title="Weekly Sales (soma)"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            strokeDash=alt.StrokeDash("Set:N"),
            tooltip=["Date:T", "Set:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
        )
        .properties(height=340, title=title)
    )
    return base_altair_style(chart)


def top_quarantine_offenders(quar_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Top N stores com mais registos em quarentena."""
    if len(quar_df) == 0 or "Store" not in quar_df.columns:
        return pd.DataFrame(columns=["Store", "Quarantine_Registos"])

    d = quar_df.copy()
    d["Store"] = pd.to_numeric(d["Store"], errors="coerce")
    d = d.dropna(subset=["Store"])

    out = (
        d.groupby("Store", as_index=False)
        .size()
        .rename(columns={"size": "Quarantine_Registos"})
        .sort_values("Quarantine_Registos", ascending=False)
        .head(top_n)
    )
    out["Store"] = out["Store"].astype(int)
    return out


def offenders_chart(offenders: pd.DataFrame, title: str):
    """Barras horizontais (ranking de offenders)."""
    if len(offenders) == 0:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=260, title=title)
        )

    chart = (
        alt.Chart(offenders)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Quarantine_Registos:Q", title=""),
            y=alt.Y("Store:N", sort="-x", title="Store"),
            color=alt.value(RED),
            tooltip=["Store:N", "Quarantine_Registos:Q"],
        )
        .properties(height=260, title=title)
    )
    return base_altair_style(chart)


# =============================================================================
# 8) HEADER (branding + legenda rápida)
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


# =============================================================================
# 9) SIDEBAR (controlos + chat)
# =============================================================================
st.sidebar.header("⚙️ Controlo")

data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"], index=0)

checks_df, valid_df, quarantine_df = load_data(data_mode)

# Junta tudo só para descobrir datas e stores disponíveis no conjunto total
df_all = pd.concat([valid_df.assign(_set="Valid"), quarantine_df.assign(_set="Quarantine")], ignore_index=True)

min_dt = df_all["Date"].min()
max_dt = df_all["Date"].max()
min_date = (min_dt.date() if pd.notna(min_dt) else datetime(2010, 1, 1).date())
max_date = (max_dt.date() if pd.notna(max_dt) else datetime(2010, 12, 31).date())

date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

store_options = sorted(df_all["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

st.sidebar.divider()

# Chat placeholder (pronto para ligares IA real depois)
show_chat = st.sidebar.toggle("💬 Chat", value=True)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Faz uma pergunta (placeholder IA)."}]

if show_chat:
    with st.sidebar.expander("💬 Chat", expanded=True):
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("Escreve aqui…")
        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            reply = "Recebido. (placeholder: aqui ligas o teu motor de IA.)"
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

# Aplica filtros em cada conjunto
valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f = apply_filters(quarantine_df, start_date, end_date, selected_stores)


# =============================================================================
# 10) TABS (Executivo / Analista / Auditor)
# =============================================================================
tab_overview, tab_sales, tab_quality, tab_insights, tab_checks = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧠 Insights", "🧱 Checks (Advanced)"]
)


# =============================================================================
# 11) OVERVIEW (visão “3 segundos”)
# =============================================================================
with tab_overview:
    total_valid = len(valid_f)
    total_quar = len(quar_f)
    total = total_valid + total_quar

    pct_valid = (total_valid / total * 100) if total else 0
    pct_quar = (total_quar / total * 100) if total else 0

    total_sales_valid = float(pd.to_numeric(valid_f.get("Weekly_Sales", pd.Series(dtype=float)), errors="coerce").sum()) if total_valid else 0.0
    total_sales_quar = float(pd.to_numeric(quar_f.get("Weekly_Sales", pd.Series(dtype=float)), errors="coerce").sum()) if total_quar else 0.0

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

    st.markdown("")

    left, right = st.columns([1, 1])

    donut_df = pd.DataFrame({"Status": ["Valid", "Quarantine"], "Count": [total_valid, total_quar]})
    donut = (
        alt.Chart(donut_df)
        .mark_arc(innerRadius=75, outerRadius=125)
        .encode(
            theta="Count:Q",
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Status:N", "Count:Q"],
        )
        .properties(height=340, title="Qualidade dos dados")
    )
    left.altair_chart(base_altair_style(donut), use_container_width=True)

    right.altair_chart(trend_chart(valid_f, quar_f, "Vendas semanais (Valid vs Quarantine)"), use_container_width=True)

    st.markdown("")
    offenders = top_quarantine_offenders(quar_f, top_n=8)
    st.altair_chart(offenders_chart(offenders, "Top Stores em Quarantine (registos)"), use_container_width=True)


# =============================================================================
# 12) VENDAS
# =============================================================================
with tab_sales:
    st.subheader("📈 Vendas")
    st.caption("Top Stores (Valid) + comparação por feriado (Valid vs Quarantine).")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        colA, colB = st.columns([1.2, 1])

        with colA:
            if len(valid_f):
                top = (
                    valid_f.groupby("Store", as_index=False)["Weekly_Sales"]
                    .sum()
                    .sort_values("Weekly_Sales", ascending=False)
                    .head(12)
                )

                bar = (
                    alt.Chart(top)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Weekly_Sales:Q", title="Weekly Sales (soma)"),
                        y=alt.Y("Store:N", sort="-x", title="Top Stores (Valid)"),
                        color=alt.value(GREEN),
                        tooltip=["Store:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
                    )
                    .properties(height=420, title="Top Stores (Valid)")
                )
                st.altair_chart(base_altair_style(bar), use_container_width=True)
            else:
                st.info("Sem dados Valid.")

        with colB:
            st.altair_chart(
                grouped_holiday_chart(valid_f, quar_f, "Feriado: média de vendas (Valid vs Quarantine)"),
                use_container_width=True
            )


# =============================================================================
# 13) QUALIDADE
# =============================================================================
with tab_quality:
    st.subheader("✅ Qualidade")
    st.caption("Todas as regras com contagem (errors e warnings) + exemplos em Quarantine.")

    if len(quar_f) == 0:
        st.success("Nada em quarentena para estes filtros.")
    else:
        err_df = parse_issue_counts(quar_f.get("__errors", pd.Series(dtype="object")))
        warn_df = parse_issue_counts(quar_f.get("__warnings", pd.Series(dtype="object")))

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🔴 Errors (todas)")
            if len(err_df):
                chart = (
                    alt.Chart(err_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Ocorrências:Q", title=""),
                        y=alt.Y("Regra:N", sort="-x", title=""),
                        color=alt.value(RED),
                        tooltip=["Regra:N", "Ocorrências:Q"],
                    )
                    .properties(height=420)
                )
                st.altair_chart(base_altair_style(chart), use_container_width=True)
                st.dataframe(err_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sem errors parseáveis.")

        with c2:
            st.markdown("### 🟠 Warnings (todas)")
            if len(warn_df):
                chart = (
                    alt.Chart(warn_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Ocorrências:Q", title=""),
                        y=alt.Y("Regra:N", sort="-x", title=""),
                        color=alt.value(AMBER),
                        tooltip=["Regra:N", "Ocorrências:Q"],
                    )
                    .properties(height=420)
                )
                st.altair_chart(base_altair_style(chart), use_container_width=True)
                st.dataframe(warn_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sem warnings parseáveis.")

        st.markdown("### 🧯 Top offenders (Quarantine por Store)")
        offenders = top_quarantine_offenders(quar_f, top_n=10)
        st.altair_chart(offenders_chart(offenders, "Stores com mais registos em Quarantine"), use_container_width=True)

        st.markdown("### 📋 Exemplos em Quarantine")
        show_cols = [
            "Store", "Date", "Weekly_Sales", "Holiday_Flag",
            "Temperature", "Fuel_Price", "CPI", "Unemployment",
            "Sales_Label", "__errors", "__warnings"
        ]
        present_cols = [c for c in show_cols if c in quar_f.columns]
        st.dataframe(quar_f[present_cols].head(80), use_container_width=True)


# =============================================================================
# 14) INSIGHTS (agora sem crash e mais “interpretável”)
# =============================================================================
with tab_insights:
    st.subheader("🧠 Insights")
    st.caption("Curva por bins (pandas) + histograma. Sem Altair transform_bin (mais robusto no Cloud).")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            maxbins = st.slider("Granularidade (bins)", 8, 40, 18, 2)
        with c2:
            agg = st.selectbox("Agregação", ["mean", "median"], index=0)
        with c3:
            show_scatter = st.toggle("Mostrar scatter + tendência", value=False)

        st.markdown("")

        features = [
            ("Temperature", "Temperatura"),
            ("Fuel_Price", "Preço do Combustível"),
            ("Unemployment", "Unemployment"),
            ("CPI", "CPI"),
        ]

        for col, label in features:
            if (col not in valid_f.columns) and (col not in quar_f.columns):
                continue

            st.markdown(f"### {label}")

            left, right = st.columns([1.3, 1])

            with left:
                st.altair_chart(
                    feature_effect_chart(
                        valid_f, quar_f, col,
                        title=f"Impacto: Weekly_Sales vs {label} (por bins)",
                        maxbins=maxbins,
                        agg=agg
                    ),
                    use_container_width=True
                )

            with right:
                st.altair_chart(
                    feature_distribution_chart(
                        valid_f, quar_f, col,
                        title=f"Distribuição: {label} (Valid vs Quarantine)",
                        maxbins=min(40, maxbins * 2)
                    ),
                    use_container_width=True
                )

            if show_scatter:
                st.altair_chart(
                    feature_scatter_trend_chart(
                        valid_f, quar_f, col,
                        title=f"Scatter + tendência: {label} vs Weekly_Sales"
                    ),
                    use_container_width=True
                )

            st.markdown("---")


# =============================================================================
# 15) CHECKS (Advanced)
# =============================================================================
with tab_checks:
    st.subheader("🧱 Checks (Advanced)")
    st.caption("Secção técnica (audit / equipa de dados).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de regras", len(checks_df))
    c2.metric("Críticas (error)", int((checks_df["criticality"] == "error").sum()) if "criticality" in checks_df.columns else 0)
    c3.metric("Avisos (warn)", int((checks_df["criticality"] == "warn").sum()) if "criticality" in checks_df.columns else 0)

    cols = [c for c in ["name", "criticality", "check", "filter", "run_config_name"] if c in checks_df.columns]
    st.dataframe(checks_df[cols], use_container_width=True)
