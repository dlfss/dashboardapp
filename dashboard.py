# app.py
# Streamlit demo: Data Quality + Walmart Weekly Sales (mock data)
# Ready to swap mock data for Databricks tables:
# - databricks_demos.sales_data.dqx_demo_walmart_checks
# - databricks_demos.sales_data.dqx_demo_walmart_valid_data
# - databricks_demos.sales_data.dqx_demo_walmart_quarantine_data

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional charts (nice + simple). If you prefer pure matplotlib, tell me.
import altair as alt


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="DQ Sales Monitor (Demo)",
    page_icon="📊",
    layout="wide",
)

st.title("📊 DQ Sales Monitor (Demo)")
st.caption(
    "Conteúdo e métricas são **fictícios** (mock). Estrutura pronta para ligar às tuas tabelas no Databricks."
)


# ----------------------------
# Utilities: mock generators
# ----------------------------
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]


@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    # Inspired by the screenshot of dqx_demo_walmart_checks
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
    # store as JSON string (mimic databricks view)
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

    stores = list(range(1, 46))  # 1..45
    start = datetime(2010, 2, 5)  # similar to screenshot
    weeks = 120  # ~2.3 years

    dates = _daterange(start, weeks)

    rows = []
    id_counter = 1

    for d in dates:
        # choose a subset of stores each week (simulate)
        active_stores = rng.choice(stores, size=rng.integers(18, 46), replace=False)
        for s in active_stores:
            base = rng.normal(1_500_000, 220_000)
            season = 1.0 + 0.10 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi)
            holiday = int(rng.random() < 0.08)  # ~8% holiday weeks
            holiday_boost = 1.0 + (0.12 if holiday else 0.0)
            weekly_sales = max(0, base * season * holiday_boost + rng.normal(0, 60_000))

            temperature = float(np.clip(rng.normal(55, 18), -10, 55))  # keep valid range
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

    # derive Sales_Label by quantiles
    q33, q66 = df["Weekly_Sales"].quantile([0.33, 0.66]).tolist()
    df["Sales_Label"] = df["Weekly_Sales"].apply(lambda v: _sales_label(v, q33, q66))

    # Now create quarantine by injecting errors into a random fraction
    df_all = df.copy()
    n = len(df_all)
    quarantine_idx = rng.choice(df_all.index, size=int(n * 0.10), replace=False)  # 10% in quarantine
    q = df_all.loc[quarantine_idx].copy()
    v = df_all.drop(quarantine_idx).copy()

    # Inject typical issues in quarantine:
    # - Temperature null or out of range
    # - Fuel_Price null
    # - Holiday_Flag invalid
    # - Sales_Label null/invalid
    checks_catalog = make_mock_checks()

    def add_issue(row: pd.Series) -> tuple[dict, dict]:
        errors = []
        warnings = []
        issue_type = rng.choice(
            ["temp_null", "temp_range", "fuel_null", "holiday_invalid", "label_null", "label_invalid"],
            p=[0.18, 0.32, 0.20, 0.10, 0.10, 0.10],
        )

        def err(name, col, msg):
            errors.append({"name": name, "column": col, "message": msg})

        def warn(name, col, msg):
            warnings.append({"name": name, "column": col, "message": msg})

        if issue_type == "temp_null":
            row["Temperature"] = None
            warn("Temperature_is_null", "Temperature", "Temperature está vazio (warning).")
        elif issue_type == "temp_range":
            row["Temperature"] = float(rng.choice([-25, 80, 120]))
            err("Temperature_isnt_in_range", "Temperature", "Temperature fora do intervalo [-10, 55].")
        elif issue_type == "fuel_null":
            row["Fuel_Price"] = None
            err("Fuel_Price_is_null", "Fuel_Price", "Fuel_Price está vazio (error).")
        elif issue_type == "holiday_invalid":
            row["Holiday_Flag"] = int(rng.choice([2, 3, -1]))
            err("Holiday_Flag_other_value", "Holiday_Flag", "Holiday_Flag fora da lista permitida {0,1}.")
        elif issue_type == "label_null":
            row["Sales_Label"] = None
            err("Sales_Label_is_null", "Sales_Label", "Sales_Label está vazio (error).")
        elif issue_type == "label_invalid":
            row["Sales_Label"] = rng.choice(["HIGH", "Med", "Unknown"])
            err("Sales_Label_other_value", "Sales_Label", "Sales_Label não está em {High, Medium, Low}.")
        return (
            {"rule_count": len(errors), "items": errors},
            {"rule_count": len(warnings), "items": warnings},
        )

    q_errors = []
    q_warnings = []
    q2 = []
    for _, r in q.iterrows():
        r = r.copy()
        e, w = add_issue(r)
        q_errors.append(json.dumps(e))
        q_warnings.append(json.dumps(w))
        q2.append(r)

    q = pd.DataFrame(q2)
    q["__errors"] = q_errors
    q["__warnings"] = q_warnings

    # valid has empty errors/warnings
    v["__errors"] = None
    v["__warnings"] = None

    return v.reset_index(drop=True), q.reset_index(drop=True)


# ----------------------------
# (Optional) Databricks loader stub
# ----------------------------
def load_from_databricks_or_mock(mode: str):
    """
    mode:
      - "mock" -> uses generated fake data
      - "databricks" -> tries to read from Spark (if running inside Databricks)
    """
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()

    if mode == "mock":
        return checks, valid_df, quarantine_df

    # Try Spark if available (Databricks)
    try:
        from pyspark.sql import SparkSession  # type: ignore

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("SparkSession not found")

        checks_s = spark.table("databricks_demos.sales_data.dqx_demo_walmart_checks")
        valid_s = spark.table("databricks_demos.sales_data.dqx_demo_walmart_valid_data")
        quar_s = spark.table("databricks_demos.sales_data.dqx_demo_walmart_quarantine_data")

        checks = checks_s.toPandas()
        valid_df = valid_s.toPandas()
        quarantine_df = quar_s.toPandas()

        return checks, valid_df, quarantine_df
    except Exception as e:
        st.warning(f"Não consegui ligar ao Databricks (a usar mock). Motivo: {e}")
        return checks, valid_df, quarantine_df


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Configuração")

data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"], index=0)
checks_df, valid_df, quarantine_df = load_from_databricks_or_mock(data_mode)

dataset_choice = st.sidebar.selectbox("Dataset", ["Valid", "Quarantine", "Ambos"], index=2)

# Common columns exist in both
df_for_filters = pd.concat([valid_df.assign(_set="Valid"), quarantine_df.assign(_set="Quarantine")], ignore_index=True)

# Ensure Date is datetime/date
df_for_filters["Date"] = pd.to_datetime(df_for_filters["Date"])

min_date = df_for_filters["Date"].min().date()
max_date = df_for_filters["Date"].max().date()

date_range = st.sidebar.date_input("Intervalo de datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

store_options = sorted(df_for_filters["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store", store_options, default=store_options[:5])

label_options = ["High", "Medium", "Low"]
selected_labels = st.sidebar.multiselect("Sales_Label", label_options, default=label_options)

holiday_options = ["0", "1", "null"]
selected_holiday = st.sidebar.multiselect("Holiday_Flag", holiday_options, default=holiday_options)

st.sidebar.divider()
st.sidebar.caption("Dica: começa com poucos Stores para ficar mais rápido e legível.")


# ----------------------------
# Filtering
# ----------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    dff["Date"] = pd.to_datetime(dff["Date"])
    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]

    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]

    # Sales_Label filter
    if selected_labels:
        dff = dff[dff["Sales_Label"].isin(selected_labels)]

    # Holiday_Flag filter (supports null)
    if selected_holiday:
        allowed = set(selected_holiday)
        mask = pd.Series(False, index=dff.index)
        if "0" in allowed:
            mask = mask | (dff["Holiday_Flag"] == 0)
        if "1" in allowed:
            mask = mask | (dff["Holiday_Flag"] == 1)
        if "null" in allowed:
            mask = mask | (dff["Holiday_Flag"].isna())
        dff = dff[mask]

    return dff


valid_f = apply_filters(valid_df)
quar_f = apply_filters(quarantine_df)

if dataset_choice == "Valid":
    base = valid_f.copy()
elif dataset_choice == "Quarantine":
    base = quar_f.copy()
else:
    base = pd.concat([valid_f.assign(_set="Valid"), quar_f.assign(_set="Quarantine")], ignore_index=True)

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_rules, tab_quarantine, tab_sales = st.tabs(
    ["📌 Overview", "🧱 Regras (Checks)", "🚧 Quarentena", "📈 Vendas"]
)


# ----------------------------
# Overview
# ----------------------------
with tab_overview:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    total_rows = len(df_for_filters)
    total_valid = len(valid_df)
    total_quar = len(quarantine_df)
    pct_valid = (total_valid / total_rows * 100) if total_rows else 0
    pct_quar = (total_quar / total_rows * 100) if total_rows else 0

    # last 7 days based on available max date
    last_date = df_for_filters["Date"].max()
    window_start = last_date - pd.Timedelta(days=7)

    quar_last7 = quarantine_df.copy()
    quar_last7["Date"] = pd.to_datetime(quar_last7["Date"])
    quar_last7 = quar_last7[quar_last7["Date"] >= window_start]

    # Parse __errors/__warnings counts if present
    def count_items(series):
        cnt = 0
        for x in series.dropna():
            try:
                payload = json.loads(x)
                cnt += len(payload.get("items", []))
            except Exception:
                pass
        return cnt

    errors_7 = count_items(quar_last7.get("__errors", pd.Series(dtype="object")))
    warns_7 = count_items(quar_last7.get("__warnings", pd.Series(dtype="object")))

    rules_total = len(checks_df)
    rules_critical = int((checks_df["criticality"] == "error").sum())
    rules_warn = int((checks_df["criticality"] == "warn").sum())

    c1.metric("Registos analisados", f"{total_rows:,}".replace(",", "."))
    c2.metric("Válidos", f"{total_valid:,}".replace(",", "."), f"{pct_valid:.1f}%")
    c3.metric("Quarentena", f"{total_quar:,}".replace(",", "."), f"{pct_quar:.1f}%")
    c4.metric("Erros (7 dias)", f"{errors_7:,}".replace(",", "."))
    c5.metric("Avisos (7 dias)", f"{warns_7:,}".replace(",", "."))
    c6.metric("Regras ativas", f"{rules_total} (crit: {rules_critical} | warn: {rules_warn})")

    st.divider()

    left, right = st.columns([1, 1])

    # Donut valid vs quarantine
    with right:
        donut_df = pd.DataFrame(
            {"set": ["Valid", "Quarantine"], "count": [total_valid, total_quar]}
        )
        donut = (
            alt.Chart(donut_df)
            .mark_arc(innerRadius=60)
            .encode(theta="count:Q", color="set:N", tooltip=["set:N", "count:Q"])
            .properties(height=260, title="Distribuição: Válidos vs Quarentena")
        )
        st.altair_chart(donut, use_container_width=True)

    # Errors by rule (from quarantine errors payload)
    with left:
        rule_counts = {}
        if "__errors" in quarantine_df.columns:
            for x in quarantine_df["__errors"].dropna():
                try:
                    payload = json.loads(x)
                    for item in payload.get("items", []):
                        name = item.get("name", "unknown")
                        rule_counts[name] = rule_counts.get(name, 0) + 1
                except Exception:
                    continue

        if rule_counts:
            rc = (
                pd.DataFrame({"rule": list(rule_counts.keys()), "count": list(rule_counts.values())})
                .sort_values("count", ascending=False)
                .head(10)
            )
            bar = (
                alt.Chart(rc)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Ocorrências"),
                    y=alt.Y("rule:N", sort="-x", title="Regra"),
                    tooltip=["rule:N", "count:Q"],
                )
                .properties(height=260, title="Top 10 regras com mais erros (quarentena)")
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Sem dados de __errors para agregação (ou dataset não tem essa coluna).")

    st.divider()
    st.subheader("Amostra (com filtros)")
    st.dataframe(base.head(30), use_container_width=True)


# ----------------------------
# Rules (Checks)
# ----------------------------
with tab_rules:
    st.subheader("🧱 Regras de Data Quality (Checks)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de regras", len(checks_df))
    c2.metric("Críticas (error)", int((checks_df["criticality"] == "error").sum()))
    c3.metric("Avisos (warn)", int((checks_df["criticality"] == "warn").sum()))

    st.write("Tabela de regras (mock / Databricks).")
    st.dataframe(checks_df, use_container_width=True)

    st.markdown("#### Detalhe de uma regra")
    rule_name = st.selectbox("Escolhe uma regra", checks_df["name"].tolist())
    rule_row = checks_df[checks_df["name"] == rule_name].iloc[0].to_dict()

    try:
        check_json = json.loads(rule_row.get("check", "{}"))
    except Exception:
        check_json = {"raw": rule_row.get("check")}

    st.json(
        {
            "name": rule_row.get("name"),
            "criticality": rule_row.get("criticality"),
            "check": check_json,
            "filter": rule_row.get("filter"),
            "run_config_name": rule_row.get("run_config_name"),
        }
    )


# ----------------------------
# Quarantine Explorer
# ----------------------------
with tab_quarantine:
    st.subheader("🚧 Quarentena — explorar erros/avisos")

    st.write(
        "Aqui consegues ver os registos que falharam regras críticas e os warnings associados. "
        "Usa filtros no sidebar."
    )

    if len(quar_f) == 0:
        st.warning("Sem registos em quarentena para estes filtros.")
    else:
        # Quick summary by rule from the filtered quarantine
        st.markdown("#### Resumo por regra (no filtro atual)")
        rule_counts_f = {}
        if "__errors" in quar_f.columns:
            for x in quar_f["__errors"].dropna():
                try:
                    payload = json.loads(x)
                    for item in payload.get("items", []):
                        name = item.get("name", "unknown")
                        rule_counts_f[name] = rule_counts_f.get(name, 0) + 1
                except Exception:
                    continue

        if rule_counts_f:
            rc_f = (
                pd.DataFrame({"rule": list(rule_counts_f.keys()), "count": list(rule_counts_f.values())})
                .sort_values("count", ascending=False)
            )
            st.dataframe(rc_f, use_container_width=True)

        st.markdown("#### Tabela de registos em quarentena")
        st.dataframe(quar_f.head(200), use_container_width=True)

        st.markdown("#### Inspecionar um registo específico")
        pick = st.number_input(
            "Linha (0 = primeira da tabela filtrada)",
            min_value=0,
            max_value=max(0, len(quar_f) - 1),
            value=0,
            step=1,
        )
        row = quar_f.iloc[int(pick)].to_dict()

        # parse __errors / __warnings
        def parse_payload(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            try:
                return json.loads(x)
            except Exception:
                return {"raw": x}

        st.json(
            {
                "Store": row.get("Store"),
                "Date": str(row.get("Date")),
                "Weekly_Sales": row.get("Weekly_Sales"),
                "Sales_Label": row.get("Sales_Label"),
                "Holiday_Flag": row.get("Holiday_Flag"),
                "Temperature": row.get("Temperature"),
                "Fuel_Price": row.get("Fuel_Price"),
                "CPI": row.get("CPI"),
                "Unemployment": row.get("Unemployment"),
                "__errors": parse_payload(row.get("__errors")),
                "__warnings": parse_payload(row.get("__warnings")),
            }
        )


# ----------------------------
# Sales (Valid data)
# ----------------------------
with tab_sales:
    st.subheader("📈 Vendas — análise (preferencialmente com dados válidos)")

    st.write(
        "Por defeito, recomenda-se usar **Valid** para métricas principais. "
        "Mesmo assim podes comparar com Quarantine no sidebar."
    )

    # Choose which dataset to chart (valid only or combined)
    chart_df = base.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])

    if len(chart_df) == 0:
        st.warning("Sem dados para os filtros selecionados.")
    else:
        # KPI block based on selection
        only_valid = valid_f.copy()
        only_valid["Date"] = pd.to_datetime(only_valid["Date"])
        total_sales_valid = float(only_valid["Weekly_Sales"].sum()) if len(only_valid) else 0.0
        avg_sales_store_week = float(only_valid.groupby(["Store", "Date"])["Weekly_Sales"].sum().mean()) if len(only_valid) else 0.0

        # holiday impact (valid only)
        if len(only_valid):
            holiday_valid = only_valid.copy()
            holiday_valid["Holiday_Flag"] = holiday_valid["Holiday_Flag"].fillna(0)
            mean_hol = holiday_valid[holiday_valid["Holiday_Flag"] == 1]["Weekly_Sales"].mean()
            mean_non = holiday_valid[holiday_valid["Holiday_Flag"] == 0]["Weekly_Sales"].mean()
            holiday_impact = ((mean_hol - mean_non) / mean_non * 100) if mean_non and not np.isnan(mean_hol) else 0.0
        else:
            holiday_impact = 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Weekly_Sales (Valid)", f"${total_sales_valid/1e9:.2f}B")
        k2.metric("Média por store/semana (Valid)", f"${avg_sales_store_week/1e6:.2f}M")
        k3.metric("Impacto médio feriados (Valid)", f"{holiday_impact:+.1f}%")

        st.divider()

        left, right = st.columns([1.2, 1])

        with left:
            # Time series: total sales per week
            ts = (
                chart_df.groupby("Date", as_index=False)["Weekly_Sales"]
                .sum()
                .sort_values("Date")
            )
            line = (
                alt.Chart(ts)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Data"),
                    y=alt.Y("Weekly_Sales:Q", title="Weekly_Sales (soma)"),
                    tooltip=["Date:T", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
                )
                .properties(height=320, title="Evolução semanal (soma de Weekly_Sales)")
            )
            st.altair_chart(line, use_container_width=True)

        with right:
            # Top stores
            top = (
                chart_df.groupby("Store", as_index=False)["Weekly_Sales"]
                .sum()
                .sort_values("Weekly_Sales", ascending=False)
                .head(10)
            )
            bar = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X("Weekly_Sales:Q", title="Weekly_Sales (soma)"),
                    y=alt.Y("Store:N", sort="-x", title="Store"),
                    tooltip=["Store:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
                )
                .properties(height=320, title="Top 10 Stores por vendas")
            )
            st.altair_chart(bar, use_container_width=True)

        st.divider()

        st.markdown("#### Tabela (amostra)")
        st.dataframe(chart_df.sort_values("Date", ascending=False).head(200), use_container_width=True)


# ----------------------------
# Footer
# ----------------------------
st.caption("✅ Demo pronto para UI. Quando quiseres, eu adapto para ler diretamente do Databricks SQL Warehouse/Connector.")

