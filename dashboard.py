# app.py — DQ Sales Monitor (Databricks-ready) + Headline automático
import json
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="DQ Sales Monitor", page_icon="📊", layout="wide")

GREEN = "#00E676"
RED = "#FF1744"
AMBER = "#FFB300"
BORDER = "rgba(255,255,255,0.10)"
PANEL_BG = "rgba(255,255,255,0.03)"

alt.themes.enable("dark")

VALID_TBL = "databricks_demos.sales_data.dqx_demo_walmart_valid_data"
QUAR_TBL = "databricks_demos.sales_data.dqx_demo_walmart_quarantine_data"
CHECKS_TBL = "databricks_demos.sales_data.dqx_demo_walmart_checks"


# ----------------------------
# CSS
# ----------------------------
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; }}
      h1, h2, h3 {{ letter-spacing: -0.02em; }}
      .small-note {{ color: rgba(255,255,255,0.65); font-size: 0.9rem; }}

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

      .panel {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.02);
      }}

      .headline {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 12px 14px;
        background: rgba(255,255,255,0.04);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        font-size: 1.02rem;
        line-height: 1.25;
      }}
      .headline b {{ font-weight: 900; }}
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Load Databricks
# ----------------------------
@st.cache_data(show_spinner=False)
def load_from_databricks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from pyspark.sql import SparkSession  # type: ignore
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("SparkSession not found. Executa isto dentro do Databricks (ou com Spark ativo).")
    valid_df = spark.table(VALID_TBL).toPandas()
    quar_df = spark.table(QUAR_TBL).toPandas()
    checks_df = spark.table(CHECKS_TBL).toPandas()
    return valid_df, quar_df, checks_df


# ----------------------------
# Helpers
# ----------------------------
def kpi_card(label: str, value: str, delta: Optional[str] = None, color: Optional[str] = None):
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


def safe_datetime_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    d = df.copy()
    if col in d.columns:
        d[col] = pd.to_datetime(d[col], errors="coerce")
    return d


def apply_filters(df: pd.DataFrame, start_date, end_date, selected_stores) -> pd.DataFrame:
    d = safe_datetime_col(df, "Date")
    if "Date" in d.columns:
        d = d[(d["Date"].dt.date >= start_date) & (d["Date"].dt.date <= end_date)]
    if selected_stores and "Store" in d.columns:
        d = d[d["Store"].isin(selected_stores)]
    return d


def parse_issue_counts(series: pd.Series) -> pd.DataFrame:
    counts = {}
    for x in series.dropna():
        try:
            payload = json.loads(x) if isinstance(x, str) else x
            items = payload.get("items", []) if isinstance(payload, dict) else []
            for item in items:
                name = item.get("name", "unknown")
                counts[name] = counts.get(name, 0) + 1
        except Exception:
            continue

    df = pd.DataFrame({"Rule": list(counts.keys()), "Count": list(counts.values())})
    if len(df):
        df = df.sort_values("Count", ascending=False).reset_index(drop=True)
    return df


def top_row(df: pd.DataFrame) -> tuple[Optional[str], int]:
    if df is None or len(df) == 0:
        return None, 0
    r = df.iloc[0]
    return str(r["Rule"]), int(r["Count"])


def compute_store_quarantine_rate(valid_f: pd.DataFrame, quar_f: pd.DataFrame) -> Optional[tuple[int, float]]:
    if "Store" not in valid_f.columns and "Store" not in quar_f.columns:
        return None

    v = valid_f[["Store"]].copy() if "Store" in valid_f.columns else pd.DataFrame({"Store": []})
    q = quar_f[["Store"]].copy() if "Store" in quar_f.columns else pd.DataFrame({"Store": []})

    v["Set"] = "Valid"
    q["Set"] = "Quarantine"
    base = pd.concat([v, q], ignore_index=True)

    if len(base) == 0:
        return None

    store_counts = base.groupby(["Store", "Set"], as_index=False).size().rename(columns={"size": "Count"})
    pivot = store_counts.pivot_table(index="Store", columns="Set", values="Count", fill_value=0).reset_index()
    pivot["Total"] = pivot.get("Valid", 0) + pivot.get("Quarantine", 0)
    pivot["QuarantineRate"] = np.where(pivot["Total"] > 0, 100.0 * pivot.get("Quarantine", 0) / pivot["Total"], 0.0)

    if len(pivot) == 0:
        return None

    worst = pivot.sort_values("QuarantineRate", ascending=False).iloc[0]
    return int(worst["Store"]), float(worst["QuarantineRate"])


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
# Sidebar: filters + Chat
# ----------------------------
st.sidebar.header("⚙️ Controlo")

valid_df, quar_df, checks_df = load_from_databricks()

all_df = pd.concat([valid_df.assign(_set="Valid"), quar_df.assign(_set="Quarantine")], ignore_index=True)
all_df = safe_datetime_col(all_df, "Date")

min_date = all_df["Date"].min().date()
max_date = all_df["Date"].max().date()

date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

store_options = sorted(all_df["Store"].dropna().unique().tolist()) if "Store" in all_df.columns else []
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

st.sidebar.divider()

# Chat panel
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

# Apply filters
valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f = apply_filters(quar_df, start_date, end_date, selected_stores)


# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_sales, tab_quality, tab_checks = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧱 Checks (Advanced)"]
)


# ----------------------------
# OVERVIEW + HEADLINE automático
# ----------------------------
with tab_overview:
    total_valid = len(valid_f)
    total_quar = len(quar_f)
    total = total_valid + total_quar

    pct_valid = (total_valid / total * 100) if total else 0
    pct_quar = (total_quar / total * 100) if total else 0

    sales_valid = float(pd.to_numeric(valid_f.get("Weekly_Sales", pd.Series(dtype="float")), errors="coerce").sum()) if total_valid else 0.0
    sales_quar = float(pd.to_numeric(quar_f.get("Weekly_Sales", pd.Series(dtype="float")), errors="coerce").sum()) if total_quar else 0.0

    # --- Headline calc ---
    err_col = "__errors" if "__errors" in quar_f.columns else "errors" if "errors" in quar_f.columns else None
    warn_col = "__warnings" if "__warnings" in quar_f.columns else "warnings" if "warnings" in quar_f.columns else None

    err_df = parse_issue_counts(quar_f[err_col]) if (len(quar_f) and err_col) else pd.DataFrame(columns=["Rule", "Count"])
    warn_df = parse_issue_counts(quar_f[warn_col]) if (len(quar_f) and warn_col) else pd.DataFrame(columns=["Rule", "Count"])

    top_err_name, top_err_count = top_row(err_df)
    top_warn_name, top_warn_count = top_row(warn_df)
    worst_store = compute_store_quarantine_rate(valid_f, quar_f)

    if total_quar == 0:
        headline_html = f"""
        <div class="headline">
          ✅ <b>OK</b> — 0% em <span class="pill pill-red">Quarantine</span>.
        </div>
        """
    else:
        store_part = ""
        if worst_store:
            store_id, store_rate = worst_store
            store_part = f" • store mais afetada: <b>{store_id}</b> ({store_rate:.1f}%)"

        err_part = f"principal erro: <b>{top_err_name}</b> ({top_err_count:,})".replace(",", ".") if top_err_name else "principal erro: <b>—</b>"
        warn_part = f"principal warning: <b>{top_warn_name}</b> ({top_warn_count:,})".replace(",", ".") if top_warn_name else "principal warning: <b>—</b>"

        headline_html = f"""
        <div class="headline">
          ⚠️ <b>Quarantine {pct_quar:.1f}%</b> — {err_part} • {warn_part}{store_part}
        </div>
        """

    st.markdown(headline_html, unsafe_allow_html=True)
    st.markdown("")

    # KPI row
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
        if total:
            ts_all = (
                pd.concat([valid_f.assign(_set="Valid"), quar_f.assign(_set="Quarantine")], ignore_index=True)
                .pipe(safe_datetime_col, "Date")
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


# ----------------------------
# VENDAS (mantém simples)
# ----------------------------
with tab_sales:
    st.subheader("📈 Vendas")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        colA, colB = st.columns([1.2, 1])

        with colA:
            if len(valid_f) and "Store" in valid_f.columns:
                top = (
                    valid_f.groupby("Store", as_index=False)["Weekly_Sales"]
                    .sum()
                    .sort_values("Weekly_Sales", ascending=False)
                    .head(12)
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
                    .properties(height=420, title="Top Stores (Valid)")
                )
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("Sem dados Valid para Top Stores.")

        with colB:
            def holiday_avg(df: pd.DataFrame, label: str) -> pd.DataFrame:
                if len(df) == 0 or "Weekly_Sales" not in df.columns:
                    return pd.DataFrame(columns=["Holiday", "Avg", "Set"])
                d = df.copy()
                d["Holiday_Flag"] = pd.to_numeric(d.get("Holiday_Flag"), errors="coerce").fillna(0)
                g = d.groupby("Holiday_Flag", as_index=False)["Weekly_Sales"].mean()
                g["Holiday"] = g["Holiday_Flag"].map({0: "Sem feriado", 1: "Com feriado"}).fillna("Outro")
                g["Set"] = label
                g = g.rename(columns={"Weekly_Sales": "Avg"})
                return g[["Holiday", "Avg", "Set"]]

            hol = pd.concat([holiday_avg(valid_f, "Valid"), holiday_avg(quar_f, "Quarantine")], ignore_index=True)

            if len(hol):
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
                    .properties(height=420, title="Impacto de feriados (Valid vs Quarantine)")
                )
                st.altair_chart(hol_bar, use_container_width=True)
            else:
                st.info("Sem dados suficientes para impacto de feriados.")


# ----------------------------
# QUALIDADE (completo)
# ----------------------------
with tab_quality:
    st.subheader("✅ Qualidade")

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
        err_col = "__errors" if "__errors" in quar_f.columns else "errors" if "errors" in quar_f.columns else None
        warn_col = "__warnings" if "__warnings" in quar_f.columns else "warnings" if "warnings" in quar_f.columns else None

        err_df = parse_issue_counts(quar_f[err_col]) if err_col else pd.DataFrame(columns=["Rule", "Count"])
        warn_df = parse_issue_counts(quar_f[warn_col]) if warn_col else pd.DataFrame(columns=["Rule", "Count"])

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

        st.markdown("### 📋 Exemplos (Quarantine)")
        show_cols = ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Sales_Label"]
        if err_col: show_cols.append(err_col)
        if warn_col: show_cols.append(warn_col)
        present_cols = [c for c in show_cols if c in quar_f.columns]
        st.dataframe(quar_f[present_cols].head(80), use_container_width=True)


# ----------------------------
# CHECKS (Advanced) — no fim, sem detalhe extra
# ----------------------------
with tab_checks:
    st.subheader("🧱 Checks (Advanced)")
    st.caption("Tabela de checks (audit / equipa técnica).")

    if len(checks_df):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de regras", len(checks_df))
        c2.metric("Críticas (error)", int((checks_df["criticality"] == "error").sum()) if "criticality" in checks_df.columns else 0)
        c3.metric("Avisos (warn)", int((checks_df["criticality"] == "warn").sum()) if "criticality" in checks_df.columns else 0)

        cols = [c for c in ["name", "criticality", "check", "filter", "run_config_name"] if c in checks_df.columns]
        st.dataframe(checks_df[cols], use_container_width=True)
    else:
        st.info("Sem dados em checks.")
