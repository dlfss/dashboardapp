# app.py
# =============================================================================
# DQ Sales Monitor (versão simples e robusta)
# =============================================================================
# Objetivo:
# - Mostrar rapidamente "Valid" vs "Quarantine"
# - KPIs + tendência + onde estão os problemas
# - Insights fáceis: bins (faixas) e distribuição
#
# Esta versão é feita para NÃO dar erro no Streamlit Cloud:
# - Sem Spark/Databricks
# - Sem Altair transform_bin (binning é feito em pandas)

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# =============================================================================
# 1) CONFIG + CORES
# =============================================================================
st.set_page_config(page_title="DQ Sales Monitor", page_icon="📊", layout="wide")

GREEN = "#00E676"
RED = "#FF1744"
AMBER = "#FFB300"


# =============================================================================
# 2) UI (CSS simples)
# =============================================================================
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.0rem; padding-bottom: 2rem; }}
      h1, h2, h3 {{ letter-spacing: -0.02em; }}

      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 800;
      }}
      .pill-green {{ background: rgba(0,230,118,0.18); color: {GREEN}; border: 1px solid rgba(0,230,118,0.55); }}
      .pill-red   {{ background: rgba(255,23,68,0.18);  color: {RED};   border: 1px solid rgba(255,23,68,0.55); }}
      .pill-amber {{ background: rgba(255,179,0,0.16);  color: {AMBER}; border: 1px solid rgba(255,179,0,0.45); }}

      .kpi {{
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 12px 14px;
        background: rgba(255,255,255,0.03);
      }}
      .kpi .label {{ font-size: 0.82rem; color: rgba(255,255,255,0.72); margin-bottom: 4px; }}
      .kpi .value {{ font-size: 1.6rem; font-weight: 900; line-height: 1.1; }}
      .kpi .sub {{ font-size: 0.9rem; color: rgba(255,255,255,0.70); margin-top: 4px; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    # 📊 DQ Sales Monitor
    <span class="pill pill-green">VALID</span>&nbsp;
    <span class="pill pill-red">QUARANTINE</span>&nbsp;
    <span class="pill pill-amber">WARNINGS</span>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 3) MOCK DATA (sempre disponível)
# =============================================================================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]


@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    rows = [
        ("Sales_Label_is_null", "error"),
        ("Sales_Label_other_value", "error"),
        ("Holiday_Flag_is_null", "warn"),
        ("Fuel_Price_is_null", "error"),
        ("Temperature_isnt_in_range", "error"),
        ("Temperature_is_null", "warn"),
        ("Unemployment_is_null", "error"),
        ("Store_is_null", "error"),
        ("Date_is_null", "error"),
        ("Weekly_Sales_is_null", "error"),
        ("Holiday_Flag_other_value", "error"),
        ("CPI_is_null", "error"),
    ]
    return pd.DataFrame(rows, columns=["name", "criticality"])


@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    "Holiday_Flag": int(holiday),
                    "Temperature": float(round(temperature, 2)),
                    "Fuel_Price": float(round(fuel_price, 3)),
                    "CPI": float(round(cpi, 3)),
                    "Unemployment": float(round(unemployment, 3)),
                }
            )
            id_counter += 1

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ~10% quarantine
    q_idx = rng.choice(df.index, size=int(len(df) * 0.10), replace=False)
    q = df.loc[q_idx].copy()
    v = df.drop(q_idx).copy()

    # Injetar issues (para aparecer na aba Qualidade)
    def add_issue(row: pd.Series):
        errors, warnings = [], []

        issue_type = rng.choice(
            ["temp_null", "temp_range", "fuel_null", "holiday_invalid"],
            p=[0.25, 0.35, 0.25, 0.15],
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

        return {"items": errors}, {"items": warnings}

    q_rows, q_err, q_warn = [], [], []
    for _, r in q.iterrows():
        r = r.copy()
        e, w = add_issue(r)
        q_rows.append(r)
        q_err.append(json.dumps(e))
        q_warn.append(json.dumps(w))

    q = pd.DataFrame(q_rows)
    q["__errors"] = q_err
    q["__warnings"] = q_warn

    v["__errors"] = None
    v["__warnings"] = None

    return v.reset_index(drop=True), q.reset_index(drop=True)


# =============================================================================
# 4) HELPERS (filtros + KPIs + parsing)
# =============================================================================
def apply_filters(df: pd.DataFrame, start_date, end_date, stores) -> pd.DataFrame:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[(d["Date"].dt.date >= start_date) & (d["Date"].dt.date <= end_date)]
    if stores:
        d = d[d["Store"].isin(stores)]
    return d


def parse_issue_counts(series: pd.Series) -> pd.DataFrame:
    counts = {}
    for x in series.dropna():
        try:
            payload = json.loads(x)
            for item in payload.get("items", []):
                name = item.get("name", "unknown")
                counts[name] = counts.get(name, 0) + 1
        except Exception:
            continue
    out = pd.DataFrame({"Regra": list(counts.keys()), "Ocorrências": list(counts.values())})
    if len(out):
        out = out.sort_values("Ocorrências", ascending=False).reset_index(drop=True)
    return out


def kpi(label: str, value: str, sub: str | None = None, color: str | None = None):
    color_style = f"color:{color};" if color else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value" style="{color_style}">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def base_style(chart):
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="rgba(255,255,255,0.80)",
            titleColor="rgba(255,255,255,0.85)",
            gridColor="rgba(255,255,255,0.08)",
        )
        .configure_legend(labelColor="rgba(255,255,255,0.85)")
    )


# =============================================================================
# 5) CHARTS (trend + donut)
# =============================================================================
def donut_quality(valid_count: int, quar_count: int):
    df = pd.DataFrame({"Status": ["Valid", "Quarantine"], "Count": [valid_count, quar_count]})
    c = (
        alt.Chart(df)
        .mark_arc(innerRadius=70, outerRadius=110)
        .encode(
            theta="Count:Q",
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Status:N", "Count:Q"],
        )
        .properties(height=280, title="Qualidade dos dados")
    )
    return base_style(c)


def trend_sales(valid_df: pd.DataFrame, quar_df: pd.DataFrame):
    data = pd.concat([valid_df.assign(Set="Valid"), quar_df.assign(Set="Quarantine")], ignore_index=True)
    if len(data) == 0:
        return base_style(alt.Chart(pd.DataFrame({"msg": ["Sem dados"]})).mark_text(size=14).encode(text="msg:N"))

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    ts = data.groupby(["Date", "Set"], as_index=False)["Weekly_Sales"].sum().sort_values("Date")

    c = (
        alt.Chart(ts)
        .mark_line(strokeWidth=4)
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
        .properties(height=280, title="Tendência (Valid vs Quarantine)")
    )
    return base_style(c)


# =============================================================================
# 6) INSIGHTS (SEM transform_bin): bins em pandas
# =============================================================================
def _prep_feature(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    d = pd.concat([valid_df.assign(Set="Valid"), quar_df.assign(Set="Quarantine")], ignore_index=True)
    if feature not in d.columns:
        return pd.DataFrame(columns=[feature, "Weekly_Sales", "Set"])
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d["Weekly_Sales"] = pd.to_numeric(d["Weekly_Sales"], errors="coerce")
    d = d.dropna(subset=[feature, "Weekly_Sales", "Set"])
    d["Set"] = pd.Categorical(d["Set"], categories=["Valid", "Quarantine"], ordered=True)
    return d


def binned_effect_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str, bins: int = 18, agg: str = "mean"):
    """
    Faz:
    - criar bins iguais para todos (valid+quarantine)
    - agregação por bin e Set
    - plot linha com pontos
    """
    data = _prep_feature(valid_df, quar_df, feature)
    if len(data) < 20:
        return base_style(alt.Chart(pd.DataFrame({"msg": ["Sem dados suficientes"]})).mark_text(size=14).encode(text="msg:N"))

    edges = np.histogram_bin_edges(data[feature].to_numpy(), bins=bins)
    if len(edges) < 3 or np.allclose(edges[0], edges[-1]):
        return base_style(alt.Chart(pd.DataFrame({"msg": ["Sem variação suficiente"]})).mark_text(size=14).encode(text="msg:N"))

    data["bin"] = pd.cut(data[feature], bins=edges, include_lowest=True)
    g = (
        data.groupby(["Set", "bin"], as_index=False)
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

    c = (
        alt.Chart(g)
        .mark_line(point=True, strokeWidth=4)
        .encode(
            x=alt.X("xmid:Q", title=feature),
            y=alt.Y("sales:Q", title=f"Weekly_Sales ({agg})"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                            legend=alt.Legend(title="", orient="bottom")),
            tooltip=[
                "Set:N",
                alt.Tooltip("n:Q", title="N", format=","),
                alt.Tooltip("bin_left:Q", title="bin left", format=".2f"),
                alt.Tooltip("bin_right:Q", title="bin right", format=".2f"),
                alt.Tooltip("sales:Q", title=f"Weekly_Sales ({agg})", format=",.0f"),
            ],
        )
        .properties(height=320, title=f"Impacto por faixas: {feature}")
    )
    return base_style(c)


def distribution_hist_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, feature: str, bins: int = 30):
    """Histograma com bins em pandas (comparação direta Valid vs Quarantine)."""
    data = _prep_feature(valid_df, quar_df, feature)
    if len(data) < 20:
        return base_style(alt.Chart(pd.DataFrame({"msg": ["Sem dados suficientes"]})).mark_text(size=14).encode(text="msg:N"))

    edges = np.histogram_bin_edges(data[feature].to_numpy(), bins=bins)
    if len(edges) < 3 or np.allclose(edges[0], edges[-1]):
        return base_style(alt.Chart(pd.DataFrame({"msg": ["Sem variação suficiente"]})).mark_text(size=14).encode(text="msg:N"))

    data["bin"] = pd.cut(data[feature], bins=edges, include_lowest=True)
    g = data.groupby(["Set", "bin"], as_index=False).size().rename(columns={"size": "count"})
    g["bin_left"] = g["bin"].apply(lambda iv: float(iv.left) if pd.notna(iv) else np.nan)
    g["bin_right"] = g["bin"].apply(lambda iv: float(iv.right) if pd.notna(iv) else np.nan)
    g["xmid"] = (g["bin_left"] + g["bin_right"]) / 2.0

    c = (
        alt.Chart(g)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("xmid:Q", title=feature),
            y=alt.Y("count:Q", title="Registos"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                            legend=alt.Legend(title="", orient="bottom")),
            tooltip=["Set:N", alt.Tooltip("count:Q", format=","), "bin_left:Q", "bin_right:Q"],
        )
        .properties(height=240, title=f"Distribuição: {feature}")
    )
    return base_style(c)


# =============================================================================
# 7) LOAD + FILTROS
# =============================================================================
checks_df = make_mock_checks()
valid_df, quarantine_df = make_mock_valid_and_quarantine()

# Datas / stores disponíveis
df_all = pd.concat([valid_df, quarantine_df], ignore_index=True)
min_dt = df_all["Date"].min()
max_dt = df_all["Date"].max()
min_date = (min_dt.date() if pd.notna(min_dt) else datetime(2010, 1, 1).date())
max_date = (max_dt.date() if pd.notna(max_dt) else datetime(2012, 12, 31).date())

st.sidebar.header("⚙️ Controlo")

date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

store_options = sorted(df_all["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

bins = st.sidebar.slider("Granularidade (bins)", 8, 40, 18, 2)
agg = st.sidebar.selectbox("Agregação (Insights)", ["mean", "median"], index=0)

valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f = apply_filters(quarantine_df, start_date, end_date, selected_stores)


# =============================================================================
# 8) TABS
# =============================================================================
tab_overview, tab_quality, tab_insights = st.tabs(["⚡ Overview", "✅ Qualidade", "🧠 Insights"])


# =============================================================================
# 9) OVERVIEW
# =============================================================================
with tab_overview:
    total_valid = len(valid_f)
    total_quar = len(quar_f)
    total = total_valid + total_quar

    pct_valid = (total_valid / total * 100) if total else 0.0
    pct_quar = (total_quar / total * 100) if total else 0.0

    sales_valid = float(pd.to_numeric(valid_f["Weekly_Sales"], errors="coerce").sum()) if total_valid else 0.0
    sales_quar = float(pd.to_numeric(quar_f["Weekly_Sales"], errors="coerce").sum()) if total_quar else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Registos", f"{total:,}".replace(",", "."))
    with c2:
        kpi("Valid", f"{pct_valid:.1f}%", sub=f"{total_valid:,} reg.".replace(",", "."), color=GREEN)
    with c3:
        kpi("Quarantine", f"{pct_quar:.1f}%", sub=f"{total_quar:,} reg.".replace(",", "."), color=RED)
    with c4:
        kpi("Vendas (Valid / Quar.)", f"${sales_valid/1e9:.2f}B", sub=f"Quar: ${sales_quar/1e9:.2f}B", color=GREEN)

    st.markdown("")
    left, right = st.columns([1, 1])
    left.altair_chart(donut_quality(total_valid, total_quar), use_container_width=True)
    right.altair_chart(trend_sales(valid_f, quar_f), use_container_width=True)


# =============================================================================
# 10) QUALIDADE
# =============================================================================
with tab_quality:
    st.subheader("✅ Qualidade (Quarantine)")

    if len(quar_f) == 0:
        st.success("Nada em quarentena para estes filtros.")
    else:
        err_df = parse_issue_counts(quar_f.get("__errors", pd.Series(dtype="object")))
        warn_df = parse_issue_counts(quar_f.get("__warnings", pd.Series(dtype="object")))

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🔴 Errors (contagem)")
            if len(err_df):
                ch = (
                    alt.Chart(err_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Ocorrências:Q", title=""),
                        y=alt.Y("Regra:N", sort="-x", title=""),
                        color=alt.value(RED),
                        tooltip=["Regra:N", "Ocorrências:Q"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(base_style(ch), use_container_width=True)
                st.dataframe(err_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sem errors parseáveis.")

        with c2:
            st.markdown("### 🟠 Warnings (contagem)")
            if len(warn_df):
                ch = (
                    alt.Chart(warn_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Ocorrências:Q", title=""),
                        y=alt.Y("Regra:N", sort="-x", title=""),
                        color=alt.value(AMBER),
                        tooltip=["Regra:N", "Ocorrências:Q"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(base_style(ch), use_container_width=True)
                st.dataframe(warn_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sem warnings parseáveis.")

        st.markdown("### 📋 Exemplos (Quarantine)")
        cols = ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "__errors", "__warnings"]
        cols = [c for c in cols if c in quar_f.columns]
        st.dataframe(quar_f[cols].head(60), use_container_width=True)


# =============================================================================
# 11) INSIGHTS (simples e legível)
# =============================================================================
with tab_insights:
    st.subheader("🧠 Insights (simples)")
    st.caption("Para cada feature: 1) impacto por faixas (bins)  2) distribuição (histograma).")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        features = ["Temperature", "Fuel_Price", "Unemployment", "CPI"]

        for feature in features:
            if feature not in valid_f.columns and feature not in quar_f.columns:
                continue

            st.markdown(f"## {feature}")

            left, right = st.columns([1.3, 1])
            with left:
                st.altair_chart(
                    binned_effect_chart(valid_f, quar_f, feature, bins=bins, agg=agg),
                    use_container_width=True
                )
            with right:
                st.altair_chart(
                    distribution_hist_chart(valid_f, quar_f, feature, bins=min(60, bins * 2)),
                    use_container_width=True
                )

            st.markdown("---")
