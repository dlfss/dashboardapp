# app.py
# Streamlit: Executive Dashboard "3 segundos"
# - Verde (Valid) / Vermelho (Quarantine) vivos
# - Overview simples e imediato
# - Quarantine aparece também
# - Errors + Warnings: lista COMPLETA com contagens
# - Chat sempre acessível no sidebar (abre/fecha)
# - Checks (Advanced) no fim, sem detalhe técnico extra

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="DQ Sales Monitor", page_icon="📊", layout="wide")

GREEN = "#00E676"   # vivid green
RED = "#FF1744"     # vivid red
AMBER = "#FFB300"   # vivid amber
PANEL_BG = "rgba(255,255,255,0.03)"
BORDER = "rgba(255,255,255,0.10)"


# ----------------------------
# CSS: mais contraste + gráficos em destaque
# ----------------------------
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; }}
      h1, h2, h3 {{ letter-spacing: -0.02em; }}

      .small-note {{ color: rgba(255,255,255,0.65); font-size: 0.9rem; }}

      /* KPI cards */
      .kpi {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px 16px;
        background: {PANEL_BG};
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

      /* Pills */
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.01em;
      }}
      .pill-green {{ background: rgba(0,230,118,0.18); color: {GREEN}; border: 1px solid rgba(0,230,118,0.55); }}
      .pill-red {{ background: rgba(255,23,68,0.18); color: {RED}; border: 1px solid rgba(255,23,68,0.55); }}
      .pill-amber {{ background: rgba(255,179,0,0.16); color: {AMBER}; border: 1px solid rgba(255,179,0,0.45); }}

      /* Panel */
      .panel {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.02);
      }}

      /* Make charts feel "bigger" */
      .stPlotlyChart, .vega-embed {{
        border-radius: 18px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# Altair: tipografia maior e mais "executivo"
alt.themes.enable("dark")


# ----------------------------
# Mock data generators
# ----------------------------
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]


@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
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
    if v >= q66:
        return "High"
    if v >= q33:
        return "Medium"
    return "Low"


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

    # quarantine ~10%
    df_all = df.copy()
    n = len(df_all)
    quarantine_idx = rng.choice(df_all.index, size=int(n * 0.10), replace=False)
    q = df_all.loc[quarantine_idx].copy()
    v = df_all.drop(quarantine_idx).copy()

    def add_issue(row: pd.Series) -> tuple[dict, dict]:
        errors = []
        warnings = []
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


def load_data(mode: str):
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()

    if mode == "mock":
        return checks, valid_df, quarantine_df

    # Try Spark if in Databricks
    try:
        from pyspark.sql import SparkSession  # type: ignore
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("SparkSession not found")

        checks = spark.table("databricks_demos.sales_data.dqx_demo_walmart_checks").toPandas()
        valid_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_valid_data").toPandas()
        quarantine_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_quarantine_data").toPandas()
        return checks, valid_df, quarantine_df
    except Exception:
        # sem warnings visuais aqui (mantemos “limpo”)
        return checks, valid_df, quarantine_df


# ----------------------------
# Helpers
# ----------------------------
def apply_filters(df: pd.DataFrame, start_date, end_date, selected_stores) -> pd.DataFrame:
    dff = df.copy()
    dff["Date"] = pd.to_datetime(dff["Date"])
    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]
    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]
    return dff


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
    df = pd.DataFrame({"Rule": list(counts.keys()), "Count": list(counts.values())})
    if len(df):
        df = df.sort_values("Count", ascending=False).reset_index(drop=True)
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


def money_short(x: float) -> str:
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    if x >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:.0f}"


# ----------------------------
# Header
# ----------------------------
st.markdown(
    f"""
    # 📊 DQ Sales Monitor
    <span class="pill pill-green">VALID</span>&nbsp;&nbsp;
    <span class="pill pill-red">QUARANTINE</span>&nbsp;&nbsp;
    <span class="pill pill-amber">WARNINGS</span>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Sidebar: filtros + Chat (toggle)
# ----------------------------
st.sidebar.header("⚙️ Controlo")
data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"], index=0)

checks_df, valid_df, quarantine_df = load_data(data_mode)

df_all = pd.concat([valid_df.assign(_set="Valid"), quarantine_df.assign(_set="Quarantine")], ignore_index=True)
df_all["Date"] = pd.to_datetime(df_all["Date"])

min_date = df_all["Date"].min().date()
max_date = df_all["Date"].max().date()

date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

store_options = sorted(df_all["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

st.sidebar.divider()

show_chat = st.sidebar.toggle("💬 Chat", value=True)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Diz-me o que queres analisar (ex.: 'porque subiu a quarentena?')."}
    ]

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

            # Placeholder “neutro” (sem textos tipo “em breve…”)
            reply = "Recebido. (placeholder: liga aqui o teu motor de IA para respostas.)"
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)


# Apply filters
valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f = apply_filters(quarantine_df, start_date, end_date, selected_stores)


# ----------------------------
# Tabs (cliente primeiro, técnico no fim)
# ----------------------------
tab_overview, tab_sales, tab_quality, tab_checks = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧱 Checks (Advanced)"]
)


# ----------------------------
# OVERVIEW: “3 segundos”
# ----------------------------
with tab_overview:
    total_valid = len(valid_f)
    total_quar = len(quar_f)
    total = total_valid + total_quar

    pct_valid = (total_valid / total * 100) if total else 0
    pct_quar = (total_quar / total * 100) if total else 0

    sales_valid = float(valid_f["Weekly_Sales"].sum()) if total_valid else 0.0
    sales_quar = float(quar_f["Weekly_Sales"].sum()) if total_quar else 0.0
    sales_total = sales_valid + sales_quar

    # KPI row (simples, imediatos)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Registos", f"{total:,}".replace(",", "."))
    with c2:
        kpi_card("Valid", f"{pct_valid:.1f}%", f"{total_valid:,} reg.".replace(",", "."), color=GREEN)
    with c3:
        kpi_card("Quarantine", f"{pct_quar:.1f}%", f"{total_quar:,} reg.".replace(",", "."), color=RED)
    with c4:
        kpi_card("Vendas (Valid)", money_short(sales_valid), color=GREEN)
    with c5:
        kpi_card("Vendas (Quarantine)", money_short(sales_quar), color=RED)

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

    with left:
        st.altair_chart(donut, use_container_width=True)

    with right:
        # Trend total (valid + quarantine) para perceber o “impacto”
        if total:
            ts_all = (
                pd.concat(
                    [valid_f.assign(_set="Valid"), quar_f.assign(_set="Quarantine")],
                    ignore_index=True
                )
                .assign(Date=lambda d: pd.to_datetime(d["Date"]))
                .groupby(["Date", "_set"], as_index=False)["Weekly_Sales"].sum()
                .sort_values("Date")
            )

            line = (
                alt.Chart(ts_all)
                .mark_line(strokeWidth=5)
                .encode(
                    x=alt.X("Date:T", title=""),
                    y=alt.Y("Weekly_Sales:Q", title="Weekly Sales (soma)"),
                    color=alt.Color(
                        "_set:N",
                        scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                        legend=alt.Legend(title="", orient="bottom"),
                    ),
                    tooltip=["Date:T", "_set:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
                )
                .properties(height=340, title="Vendas semanais (Valid vs Quarantine)")
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("Sem dados para o filtro atual.")

    # Mini bloco “o que está mal”
    if total_quar == 0:
        st.markdown(
            f"""
            <div class="panel">
              <b>Estado:</b> <span class="pill pill-green">OK</span> — não existem registos em <span class="pill pill-red">Quarantine</span>.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        err_df = parse_issue_counts(quar_f.get("__errors", pd.Series(dtype="object")))
        warn_df = parse_issue_counts(quar_f.get("__warnings", pd.Series(dtype="object")))
        err_total = int(err_df["Count"].sum()) if len(err_df) else 0
        warn_total = int(warn_df["Count"].sum()) if len(warn_df) else 0

        st.markdown(
            f"""
            <div class="panel">
              <b>Estado:</b> <span class="pill pill-red">ATENÇÃO</span>
              &nbsp;•&nbsp; Registos em quarentena: <b>{pct_quar:.1f}%</b>
              &nbsp;•&nbsp; Errors: <b style="color:{RED}">{err_total}</b>
              &nbsp;•&nbsp; Warnings: <b style="color:{AMBER}">{warn_total}</b>
              <div class="small-note">Vai à aba <b>Qualidade</b> para veres as regras e contagens completas.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ----------------------------
# VENDAS: cliente vê performance (mas sem esconder o efeito)
# ----------------------------
with tab_sales:
    st.subheader("📈 Vendas (fácil)")
    st.caption("Comparação rápida entre Valid e Quarantine.")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        colA, colB = st.columns([1.2, 1])

        # Top stores (Valid)
        if len(valid_f):
            top = (
                valid_f.groupby("Store", as_index=False)["Weekly_Sales"]
                .sum()
                .sort_values("Weekly_Sales", ascending=False)
                .head(10)
            )
            bar = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X("Weekly_Sales:Q", title="Weekly Sales (soma)"),
                    y=alt.Y("Store:N", sort="-x", title="Top Stores (Valid)"),
                    color=alt.value(GREEN),
                    tooltip=["Store:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
                )
                .properties(height=400, title="Top 10 Stores (Valid)")
            )
        else:
            bar = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

        # Impacto feriados (Valid vs Quarantine)
        def holiday_avg(df: pd.DataFrame, label: str) -> pd.DataFrame:
            if len(df) == 0:
                return pd.DataFrame(columns=["Holiday", "Avg", "Set"])
            d = df.copy()
            d["Holiday_Flag"] = d["Holiday_Flag"].fillna(0)
            g = d.groupby("Holiday_Flag", as_index=False)["Weekly_Sales"].mean()
            g["Holiday"] = g["Holiday_Flag"].map({0: "Sem feriado", 1: "Com feriado"})
            g["Set"] = label
            g = g.rename(columns={"Weekly_Sales": "Avg"})
            return g[["Holiday", "Avg", "Set"]]

        hol = pd.concat([holiday_avg(valid_f, "Valid"), holiday_avg(quar_f, "Quarantine")], ignore_index=True)

        hol_bar = (
            alt.Chart(hol)
            .mark_bar()
            .encode(
                x=alt.X("Holiday:N", title=""),
                y=alt.Y("Avg:Q", title="Média Weekly Sales"),
                color=alt.Color(
                    "Set:N",
                    scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                    legend=alt.Legend(title="", orient="bottom"),
                ),
                tooltip=["Set:N", "Holiday:N", alt.Tooltip("Avg:Q", format=",.0f")],
            )
            .properties(height=400, title="Impacto médio de feriados (Valid vs Quarantine)")
        )

        with colA:
            if len(valid_f):
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("Sem dados Valid para mostrar Top Stores.")
        with colB:
            if len(hol):
                st.altair_chart(hol_bar, use_container_width=True)
            else:
                st.info("Sem dados para calcular impacto de feriados.")

        st.markdown("#### Amostra (Valid)")
        if len(valid_f):
            st.dataframe(
                valid_f[["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Sales_Label"]].head(50),
                use_container_width=True
            )
        else:
            st.info("Sem amostra Valid para estes filtros.")


# ----------------------------
# QUALIDADE: completo, mas bem apresentado
# ----------------------------
with tab_quality:
    st.subheader("✅ Qualidade (o que está a falhar)")
    st.caption("Lista completa de regras + quantas vezes aconteceram (Errors e Warnings).")

    if len(quar_f) == 0:
        st.markdown(
            f"""
            <div class="panel">
              <b>Estado:</b> <span class="pill pill-green">OK</span> — nada em <span class="pill pill-red">Quarantine</span>.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        err_df = parse_issue_counts(quar_f.get("__errors", pd.Series(dtype="object")))
        warn_df = parse_issue_counts(quar_f.get("__warnings", pd.Series(dtype="object")))

        err_total = int(err_df["Count"].sum()) if len(err_df) else 0
        warn_total = int(warn_df["Count"].sum()) if len(warn_df) else 0

        k1, k2, k3 = st.columns(3)
        with k1:
            kpi_card("Registos Quarantine", f"{len(quar_f):,}".replace(",", "."), color=RED)
        with k2:
            kpi_card("Total Errors", f"{err_total:,}".replace(",", "."), color=RED)
        with k3:
            kpi_card("Total Warnings", f"{warn_total:,}".replace(",", "."), color=AMBER)

        st.markdown("")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🔴 Errors (todas as regras)")
            if len(err_df):
                err_chart = (
                    alt.Chart(err_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title=""),
                        y=alt.Y("Rule:N", sort="-x", title=""),
                        color=alt.value(RED),
                        tooltip=["Rule:N", "Count:Q"],
                    )
                    .properties(height=420)
                )
                st.altair_chart(err_chart, use_container_width=True)
                st.dataframe(err_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sem errors parseáveis.")

        with c2:
            st.markdown("### 🟠 Warnings (todas as regras)")
            if len(warn_df):
                warn_chart = (
                    alt.Chart(warn_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title=""),
                        y=alt.Y("Rule:N", sort="-x", title=""),
                        color=alt.value(AMBER),
                        tooltip=["Rule:N", "Count:Q"],
                    )
                    .properties(height=420)
                )
                st.altair_chart(warn_chart, use_container_width=True)
                st.dataframe(warn_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sem warnings parseáveis.")

        st.markdown("### 📋 Exemplos em Quarantine")
        show_cols = ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "Sales_Label", "__errors", "__warnings"]
        present_cols = [c for c in show_cols if c in quar_f.columns]
        st.dataframe(quar_f[present_cols].head(80), use_container_width=True)


# ----------------------------
# CHECKS (Advanced) - último, sem “detalhe”
# ----------------------------
with tab_checks:
    st.subheader("🧱 Checks (Advanced)")
    st.caption("Secção técnica (auditoria / equipa de dados).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de regras", len(checks_df))
    c2.metric("Críticas (error)", int((checks_df["criticality"] == "error").sum()))
    c3.metric("Avisos (warn)", int((checks_df["criticality"] == "warn").sum()))

    st.dataframe(checks_df[["name", "criticality", "check"]], use_container_width=True)

