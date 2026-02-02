# app.py
# DQ Sales Monitor — Visual "3 segundos"
# Objetivo do dashboard:
# - Mostrar, em poucos segundos, a diferença entre dados "Valid" e "Quarantine"
# - Dar indicadores rápidos (KPIs), tendência temporal e onde estão os problemas
# - Permitir filtros simples (datas + store)
# - Dar uma visão executiva (Overview) + visão analítica (Qualidade/Insights) + audit trail (Checks)

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# =============================================================================
# 1) CONFIGURAÇÃO GLOBAL (Streamlit + tema + paleta)
# =============================================================================
# Define layout wide para caber mais informação sem scroll horizontal
st.set_page_config(page_title="DQ Sales Monitor", page_icon="📊", layout="wide")

# Paleta "vivid" para leitura instantânea ("3 segundos"):
# - verde = bom / válido
# - vermelho = quarentena / crítico
# - âmbar = warnings (atenção, mas não necessariamente falha dura)
GREEN = "#00E676"
RED = "#FF1744"
AMBER = "#FFB300"

# Altair tem suporte a temas; o "dark" pode variar por versão.
# Como tu já aplicas um estilo dark manual via base_altair_style(),
# isto é "nice to have". Se a versão do Altair não suportar, não crasha.
try:
    alt.themes.enable("dark")
except Exception:
    pass


# =============================================================================
# 2) CSS / UI (cards KPI, pílulas e look premium)
# =============================================================================
# Aqui estás a "hackear" um pouco a UI do Streamlit via CSS:
# - cards KPI (bordas, blur, sombra)
# - pills coloridas (valid/quarantine/warnings)
# Isto ajuda MUITO o objetivo "executivo", porque fica mais "produto" e menos "demo".
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
      .pill-red {{ background: rgba(255,23,68,0.18); color: {RED}; border: 1px solid rgba(255,23,68,0.55); }}
      .pill-amber {{ background: rgba(255,179,0,0.16); color: {AMBER}; border: 1px solid rgba(255,179,0,0.45); }}

      .panel {{
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.02);
      }}
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 3) MOCK DATA (para correr em qualquer sítio sem Databricks)
# =============================================================================
# Este bloco é o teu "modo demo / modo cloud":
# - Streamlit Cloud normalmente não tem Spark/Databricks
# - Portanto geras dados realistas, com patterns de erros/warnings

def _daterange(start: datetime, weeks: int) -> list[datetime]:
    """
    Cria uma lista de datas semanais.
    Usado para simular uma série temporal (ex: 120 semanas).
    """
    return [start + timedelta(days=7 * i) for i in range(weeks)]


@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    """
    Simula a tabela de checks/regras do pipeline DQ.

    Output:
    - DataFrame com colunas semelhantes ao "databricks_demos...checks":
      name, criticality, check, filter, run_config_name

    Nota:
    - "check" é armazenado como JSON em string, simulando o formato real.
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

    # Converte o objeto python dict -> string JSON,
    # para ficar igual ao que vem de sistemas reais.
    df["check"] = df["check"].apply(lambda x: json.dumps(x))
    return df


def _sales_label(v: float, q33: float, q66: float) -> str:
    """
    Etiqueta High/Medium/Low baseada em tercis do Weekly_Sales:
    - >= q66 -> High
    - >= q33 -> Medium
    - else -> Low
    """
    if v >= q66:
        return "High"
    if v >= q33:
        return "Medium"
    return "Low"


@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera dataset mock com:
    - valid (~90%): dados limpos
    - quarantine (~10%): dados com issues e logs

    Estrutura:
    - Colunas negócio: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Sales_Label
    - Colunas DQ: __errors e __warnings (string JSON com lista de items)
    """
    rng = np.random.default_rng(seed)

    # stores 1..45 (parecido ao dataset walmart)
    stores = list(range(1, 46))

    # Começa numa data realista e simula ~120 semanas
    start = datetime(2010, 2, 5)
    weeks = 120
    dates = _daterange(start, weeks)

    rows = []
    id_counter = 1

    # Geração de dados:
    # - base de vendas ~ Normal(1.5M, 220k)
    # - sazonalidade via sin(2*pi*dia/365)
    # - feriado em ~8% das semanas com boost ~+12%
    for d in dates:
        # Simula que nem todas as lojas estão sempre ativas (25..45 lojas/semana)
        active_stores = rng.choice(stores, size=rng.integers(25, 46), replace=False)

        for s in active_stores:
            base = rng.normal(1_500_000, 220_000)

            season = 1.0 + 0.10 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi)

            holiday = int(rng.random() < 0.08)
            holiday_boost = 1.0 + (0.12 if holiday else 0.0)

            weekly_sales = max(0, base * season * holiday_boost + rng.normal(0, 60_000))

            # Features auxiliares:
            # Nota: já "clipas" os valores aqui para ficarem plausíveis em valid
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

    # Calcula tercis e etiqueta as vendas
    q33, q66 = df["Weekly_Sales"].quantile([0.33, 0.66]).tolist()
    df["Sales_Label"] = df["Weekly_Sales"].apply(lambda v: _sales_label(v, q33, q66))

    # Define o subset quarentena (~10% aleatório).
    # Isto simula pipeline DQ que separa registos inválidos.
    df_all = df.copy()
    n = len(df_all)
    quarantine_idx = rng.choice(df_all.index, size=int(n * 0.10), replace=False)
    q = df_all.loc[quarantine_idx].copy()
    v = df_all.drop(quarantine_idx).copy()

    # Função local que injeta um tipo de problema num registo:
    # - alguns são warnings (ex: Temperature null)
    # - outros são errors (ex: Fuel_Price null, Holiday inválido, etc.)
    def add_issue(row: pd.Series) -> tuple[dict, dict]:
        errors = []
        warnings = []

        # Escolha do tipo de issue com probabilidades definidas
        issue_type = rng.choice(
            ["temp_null", "temp_range", "fuel_null", "holiday_invalid", "label_null", "label_invalid"],
            p=[0.18, 0.34, 0.20, 0.10, 0.08, 0.10],
        )

        # Helpers para adicionar items aos logs
        def err(name, col, msg):
            errors.append({"name": name, "column": col, "message": msg})

        def warn(name, col, msg):
            warnings.append({"name": name, "column": col, "message": msg})

        # Cada caso altera o row e grava no JSON de errors/warnings
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

    # Aplica issues a cada linha em quarantine
    q_errors, q_warnings, q2 = [], [], []
    for _, r in q.iterrows():
        r = r.copy()
        e, w = add_issue(r)
        q_errors.append(json.dumps(e))
        q_warnings.append(json.dumps(w))
        q2.append(r)

    q = pd.DataFrame(q2)

    # Campos com JSON (string). O teu parse_issue_counts lê isto mais tarde.
    q["__errors"] = q_errors
    q["__warnings"] = q_warnings

    # valid não tem issues, logo None
    v["__errors"] = None
    v["__warnings"] = None

    return v.reset_index(drop=True), q.reset_index(drop=True)


# =============================================================================
# 4) Normalização + Carregamento de dados (Mock vs Databricks)
# =============================================================================
def _normalize_dates_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza Date para datetime64 uma vez (boas práticas):
    - evita reconverter no apply_filters para cada filtro/tabela/plot
    - reduz CPU em datasets maiores (Databricks real)
    """
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_data(mode: str):
    """
    Carrega checks + valid_df + quarantine_df.

    Estratégia:
    - Sempre inicializa com mock (para ter fallback seguro)
    - Se mode == databricks: tenta Spark
    - Se falhar Spark: devolve mock (sem crash)
    """
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()

    if mode == "mock":
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df

    try:
        from pyspark.sql import SparkSession  # type: ignore

        # Nota: getActiveSession() depende do ambiente Databricks
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("SparkSession not found")

        # Lê tabelas demo
        checks = spark.table("databricks_demos.sales_data.dqx_demo_walmart_checks").toPandas()
        valid_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_valid_data").toPandas()
        quarantine_df = spark.table("databricks_demos.sales_data.dqx_demo_walmart_quarantine_data").toPandas()

        # Normaliza Date (importante para filtros/plots)
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)

        return checks, valid_df, quarantine_df

    except Exception:
        # Fallback silencioso para mock
        _normalize_dates_inplace(valid_df)
        _normalize_dates_inplace(quarantine_df)
        return checks, valid_df, quarantine_df


# =============================================================================
# 5) Helpers (filtros + parsing de errors/warnings + UI KPI + estilo Altair)
# =============================================================================
def apply_filters(df: pd.DataFrame, start_date, end_date, selected_stores) -> pd.DataFrame:
    """
    Aplica filtros de UI em qualquer dataframe (valid/quarantine):
    - converte Date se necessário
    - filtra por intervalo de datas (comparando date() para evitar timezone issues)
    - filtra por stores se o user selecionar alguns

    Nota:
    - dff["Date"].dt.date é prático, mas pode ser mais lento em datasets enormes.
      Aqui está ok porque tu já normalizas e o objetivo é dashboard, não ETL.
    """
    dff = df.copy()

    # Proteção extra: se Date não for datetime por alguma razão, converte.
    if "Date" in dff.columns and not np.issubdtype(dff["Date"].dtype, np.datetime64):
        dff["Date"] = pd.to_datetime(dff["Date"], errors="coerce")

    # Filtra intervalo (inclusive)
    dff = dff[(dff["Date"].dt.date >= start_date) & (dff["Date"].dt.date <= end_date)]

    # Filtro opcional por Store
    if selected_stores:
        dff = dff[dff["Store"].isin(selected_stores)]

    return dff


def parse_issue_counts(series: pd.Series) -> pd.DataFrame:
    """
    Conta ocorrências de regras (issues) a partir de uma coluna JSON string.

    Entrada típica:
    series = quar_f["__errors"] ou quar_f["__warnings"]
    onde cada célula tem:
      {"items":[{"name":"...", "column":"...", "message":"..."}, ...]}

    Output:
    - DataFrame com colunas:
      Regra / Ocorrências
    - ordenado desc por ocorrências
    """
    counts = {}

    for x in series.dropna():
        try:
            payload = json.loads(x)
            for item in payload.get("items", []):
                name = item.get("name", "unknown")
                counts[name] = counts.get(name, 0) + 1
        except Exception:
            # Se JSON estiver malformado ou célula estranha, ignora e segue.
            continue

    df = pd.DataFrame({"Regra": list(counts.keys()), "Ocorrências": list(counts.values())})
    if len(df):
        df = df.sort_values("Ocorrências", ascending=False).reset_index(drop=True)
    return df


def kpi_card(label: str, value: str, delta: str | None = None, color: str | None = None):
    """
    Renderiza um KPI em HTML para ficar "premium".
    - label = título do KPI
    - value = valor principal (grande)
    - delta = texto menor (ex: número absoluto de registos)
    - color = cor opcional do value
    """
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
    """
    Estilo base consistente para todos os charts:
    - remove borda da view (strokeOpacity=0)
    - ajusta cores de eixos / legendas / grids para dark UI
    - mantém o look uniforme em todo o dashboard
    """
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
# 6) INSIGHTS: zone bins (Baixa/Média/Alta) com fallback + gráficos agrupados
# =============================================================================
def zone_bins(df: pd.DataFrame, xcol: str, set_label: str) -> pd.DataFrame:
    """
    Transforma um feature contínuo (Temperature/Fuel/CPI/Unemployment) em 3 zonas:
    - Baixa / Média / Alta
    e calcula a média de Weekly_Sales por zona.

    Porquê isto é útil:
    - permite insight simples (executivo) sem regressão/estatística pesada
    - visual "3 segundos": barras por zona

    Robustez:
    - tenta pd.qcut em tercis
    - se falhar (pouca variabilidade / duplicates), fallback para quantis manuais
    - se dataset muito pequeno (<5 linhas após dropna), retorna vazio
    """
    if len(df) == 0 or xcol not in df.columns or "Weekly_Sales" not in df.columns:
        return pd.DataFrame(columns=["Zona", "AvgSales", "Set"])

    d = df[[xcol, "Weekly_Sales"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d["Weekly_Sales"] = pd.to_numeric(d["Weekly_Sales"], errors="coerce")
    d = d.dropna()

    if len(d) < 5:
        return pd.DataFrame(columns=["Zona", "AvgSales", "Set"])

    # 1) tentativa de tercis automáticos (qcut)
    try:
        d["Zona"] = pd.qcut(d[xcol], q=3, labels=["Baixa", "Média", "Alta"], duplicates="drop")
    except Exception:
        d["Zona"] = None

    # 2) fallback: quantis simples
    if d["Zona"].isna().all():
        q1 = d[xcol].quantile(0.33)
        q2 = d[xcol].quantile(0.66)

        def _zone(v):
            if v <= q1:
                return "Baixa"
            if v <= q2:
                return "Média"
            return "Alta"

        d["Zona"] = d[xcol].apply(_zone)

    # Agrega por zona -> média de vendas
    g = d.groupby("Zona", as_index=False)["Weekly_Sales"].mean()
    g = g.rename(columns={"Weekly_Sales": "AvgSales"})
    g["Set"] = set_label

    # Converte para categorical para garantir ordem fixa (não aleatória)
    g["Zona"] = pd.Categorical(g["Zona"], categories=["Baixa", "Média", "Alta"], ordered=True)
    g["Set"] = pd.Categorical(g["Set"], categories=["Valid", "Quarantine"], ordered=True)
    return g


def grouped_zone_chart(valid_zone: pd.DataFrame, quar_zone: pd.DataFrame, title: str):
    """
    Constrói um único gráfico com barras lado a lado por zona:
      Zona (x) -> AvgSales (y)
      xOffset = Set (Valid vs Quarantine)

    Ponto importante:
    - Mesmo que não existam dados num set por causa de filtros,
      tu queres manter as categorias/legenda consistentes.
    - Por isso crias um "template" com todas as combinações (Zona x Set),
      e depois fazes concat + groupby.

    Resultado:
    - UI estável: nunca "desaparece" uma categoria/legenda.
    """
    template = pd.DataFrame(
        [(z, s, np.nan) for z in ["Baixa", "Média", "Alta"] for s in ["Valid", "Quarantine"]],
        columns=["Zona", "Set", "AvgSales"]
    )

    data = pd.concat([template, valid_zone, quar_zone], ignore_index=True)

    # Garante 1 valor por (Zona, Set)
    data = data.groupby(["Zona", "Set"], as_index=False)["AvgSales"].mean()

    data["Zona"] = pd.Categorical(data["Zona"], categories=["Baixa", "Média", "Alta"], ordered=True)
    data["Set"] = pd.Categorical(data["Set"], categories=["Valid", "Quarantine"], ordered=True)

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Zona:N", sort=["Baixa", "Média", "Alta"], title=""),
            xOffset=alt.XOffset("Set:N"),
            y=alt.Y("AvgSales:Q", title="Média Weekly Sales"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Set:N", "Zona:N", alt.Tooltip("AvgSales:Q", format=",.0f")],
        )
        .properties(height=360, title=title)
    )

    return base_altair_style(chart)


def grouped_holiday_chart(valid_df: pd.DataFrame, quar_df: pd.DataFrame, title: str):
    """
    Compara média de vendas em "Com feriado" vs "Sem feriado".
    - Calcula média por Holiday_Flag (0/1) em cada set
    - Mostra barras agrupadas (Valid vs Quarantine)

    Robustez:
    - Remove Holiday_Flag inválidos (2, -1, etc.) para não gerar categoria estranha.
    """
    def holiday_avg(df: pd.DataFrame, set_label: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=["Holiday", "AvgSales", "Set"])

        d = df.copy()
        d["Holiday_Flag"] = pd.to_numeric(d.get("Holiday_Flag"), errors="coerce")

        # Mantém apenas 0/1 para "limpar" outliers
        d = d[d["Holiday_Flag"].isin([0, 1])]
        if len(d) == 0:
            return pd.DataFrame(columns=["Holiday", "AvgSales", "Set"])

        g = d.groupby("Holiday_Flag", as_index=False)["Weekly_Sales"].mean()
        g["Holiday"] = g["Holiday_Flag"].map({0: "Sem feriado", 1: "Com feriado"})
        g = g.rename(columns={"Weekly_Sales": "AvgSales"})
        g["Set"] = set_label
        return g[["Holiday", "AvgSales", "Set"]]

    data = pd.concat(
        [holiday_avg(valid_df, "Valid"), holiday_avg(quar_df, "Quarantine")],
        ignore_index=True
    )

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
    """
    Mostra a evolução das vendas no tempo:
    - soma Weekly_Sales por Date e Set
    - plot em linhas com dash diferente por Set

    Porque soma e não média:
    - soma representa "volume total" e cria contraste maior (bom para executivo)
    - mas poderias trocar por média para ver comportamento por store/registo
    """
    data = pd.concat([valid_df.assign(Set="Valid"), quar_df.assign(Set="Quarantine")], ignore_index=True)

    if len(data) == 0:
        return (
            alt.Chart(pd.DataFrame({"msg": ["Sem dados"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=340, title=title)
        )

    # Proteção: garante datetime
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # Agrega por semana (Date) e Set
    ts = (
        data.groupby(["Date", "Set"], as_index=False)["Weekly_Sales"].sum()
        .sort_values("Date")
    )

    # Garante ordem na legenda
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
            # Usa dash para diferenciar sem tapar
            strokeDash=alt.StrokeDash("Set:N"),
            tooltip=["Date:T", "Set:N", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
        )
        .properties(height=340, title=title)
    )

    return base_altair_style(chart)


# =============================================================================
# 7) "TOP OFFENDERS" (Quarantine por Store) - highlight rápido
# =============================================================================
def top_quarantine_offenders(quar_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Objetivo: mostrar rapidamente onde está a maior concentração de quarentena.
    Isto é um "insight executivo":
    - Que stores estão a gerar mais registos inválidos?

    Nota:
    - Aqui estás a contar registos, não severity.
      Se quisesses severity, poderias pesar errors > warnings.
    """
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
    """
    Gráfico de barras horizontal:
    - eixo x = contagem
    - eixo y = store
    """
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
# 8) HEADER (título + pills)
# =============================================================================
# Isto é apenas branding/legenda rápida: o user entende a cor sem ler texto.
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

# Toggle de fonte:
# - mock: corre sempre
# - databricks: tenta spark (se falhar, fallback para mock)
data_mode = st.sidebar.radio("Fonte de dados", ["mock", "databricks"], index=0)

# Carrega datasets (já com Date normalizado)
checks_df, valid_df, quarantine_df = load_data(data_mode)

# df_all é usado para:
# - descobrir min/max datas globais
# - descobrir lista de stores
df_all = pd.concat([valid_df.assign(_set="Valid"), quarantine_df.assign(_set="Quarantine")], ignore_index=True)

# Proteções se as datas vierem vazias (NaT)
min_dt = df_all["Date"].min()
max_dt = df_all["Date"].max()
min_date = (min_dt.date() if pd.notna(min_dt) else datetime(2010, 1, 1).date())
max_date = (max_dt.date() if pd.notna(max_dt) else datetime(2010, 12, 31).date())

# date_input devolve tuple quando é range; se user mexer errado, tens fallback
date_range = st.sidebar.date_input("Datas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Stores possíveis
store_options = sorted(df_all["Store"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect("Store (opcional)", store_options, default=[])

st.sidebar.divider()

# Chat placeholder:
# - toggle para ligar/desligar (para não ocupar espaço)
# - expander para ter chat sempre acessível
# - session_state para histórico
show_chat = st.sidebar.toggle("💬 Chat", value=True)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Faz uma pergunta (placeholder IA)."}]

if show_chat:
    with st.sidebar.expander("💬 Chat", expanded=True):
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # st.chat_input em sidebar pode ter quirks dependendo da versão do Streamlit.
        # Se um dia tiveres bugs, troca por text_input + button.
        user_msg = st.chat_input("Escreve aqui…")
        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # Placeholder: aqui ligas um motor real (OpenAI/Groq/OpenRouter/etc.)
            reply = "Recebido. (placeholder: aqui ligas o teu motor de IA.)"
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

# Aplica filtros
valid_f = apply_filters(valid_df, start_date, end_date, selected_stores)
quar_f = apply_filters(quarantine_df, start_date, end_date, selected_stores)


# =============================================================================
# 10) TABS (separação de público: executivo vs analista vs auditor)
# =============================================================================
tab_overview, tab_sales, tab_quality, tab_insights, tab_checks = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧠 Insights", "🧱 Checks (Advanced)"]
)


# =============================================================================
# 11) OVERVIEW (o ecrã "3 segundos")
# =============================================================================
with tab_overview:
    # KPIs principais:
    # - total registos
    # - % valid / % quarantine
    # - total de vendas em cada set
    total_valid = len(valid_f)
    total_quar = len(quar_f)
    total = total_valid + total_quar

    pct_valid = (total_valid / total * 100) if total else 0
    pct_quar = (total_quar / total * 100) if total else 0

    # Soma de vendas (com to_numeric para proteger strings)
    total_sales_valid = float(pd.to_numeric(valid_f.get("Weekly_Sales", pd.Series(dtype=float)), errors="coerce").sum()) if total_valid else 0.0
    total_sales_quar = float(pd.to_numeric(quar_f.get("Weekly_Sales", pd.Series(dtype=float)), errors="coerce").sum()) if total_quar else 0.0

    # Linha de KPIs (5 colunas)
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

    # Donut + Trend lado-a-lado
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

    # “Top offenders” logo no overview = insight instantâneo
    st.markdown("")
    offenders = top_quarantine_offenders(quar_f, top_n=8)
    st.altair_chart(offenders_chart(offenders, "Top Stores em Quarantine (registos)"), use_container_width=True)


# =============================================================================
# 12) VENDAS (métricas de performance)
# =============================================================================
with tab_sales:
    st.subheader("📈 Vendas")
    st.caption("Top Stores (Valid) + comparação por feriado (Valid vs Quarantine).")

    # Se não houver dados em ambos sets, avisar
    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        colA, colB = st.columns([1.2, 1])

        # A) Top Stores (apenas Valid)
        # Justificação:
        # - para performance, usa dados limpos para decisões (evitar distorções)
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

        # B) Comparação por feriado (Valid vs Quarantine)
        with colB:
            st.altair_chart(
                grouped_holiday_chart(valid_f, quar_f, "Feriado: média de vendas (Valid vs Quarantine)"),
                use_container_width=True
            )


# =============================================================================
# 13) QUALIDADE (onde estão os problemas e quantos)
# =============================================================================
with tab_quality:
    st.subheader("✅ Qualidade")
    st.caption("Todas as regras com contagem (errors e warnings).")

    if len(quar_f) == 0:
        # Se não há quarentena, é sinal "verde"
        st.success("Nada em quarentena para estes filtros.")
    else:
        # Constrói tabelas agregadas por nome de regra
        err_df = parse_issue_counts(quar_f.get("__errors", pd.Series(dtype="object")))
        warn_df = parse_issue_counts(quar_f.get("__warnings", pd.Series(dtype="object")))

        # Layout 2 colunas: Errors vs Warnings
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

                # Tabela (para equipa de dados copiar/filtrar)
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

        # "Top offenders" também aqui faz sentido (qualidade operacional)
        st.markdown("### 🧯 Top offenders (Quarantine por Store)")
        offenders = top_quarantine_offenders(quar_f, top_n=10)
        st.altair_chart(offenders_chart(offenders, "Stores com mais registos em Quarantine"), use_container_width=True)

        # Exemplos de registos quarentena (debug operacional)
        st.markdown("### 📋 Exemplos em Quarantine")
        show_cols = [
            "Store", "Date", "Weekly_Sales", "Holiday_Flag",
            "Temperature", "Fuel_Price", "CPI", "Unemployment",
            "Sales_Label", "__errors", "__warnings"
        ]
        present_cols = [c for c in show_cols if c in quar_f.columns]
        st.dataframe(quar_f[present_cols].head(80), use_container_width=True)


# =============================================================================
# 14) INSIGHTS (correlações simples por zonas)
# =============================================================================
with tab_insights:
    st.subheader("🧠 Insights")
    st.caption("Zonas Baixa / Média / Alta — barras lado a lado (Valid vs Quarantine).")

    if len(valid_f) == 0 and len(quar_f) == 0:
        st.warning("Sem dados para estes filtros.")
    else:
        # Para cada feature:
        # 1) binning em 3 zonas
        # 2) chart com barras agrupadas
        v = zone_bins(valid_f, "Temperature", "Valid")
        q = zone_bins(quar_f, "Temperature", "Quarantine")
        st.altair_chart(grouped_zone_chart(v, q, "Vendas vs Temperatura (zonas)"), use_container_width=True)

        v = zone_bins(valid_f, "Fuel_Price", "Valid")
        q = zone_bins(quar_f, "Fuel_Price", "Quarantine")
        st.altair_chart(grouped_zone_chart(v, q, "Vendas vs Preço do Combustível (zonas)"), use_container_width=True)

        v = zone_bins(valid_f, "Unemployment", "Valid")
        q = zone_bins(quar_f, "Unemployment", "Quarantine")
        st.altair_chart(grouped_zone_chart(v, q, "Vendas vs Unemployment (zonas)"), use_container_width=True)

        v = zone_bins(valid_f, "CPI", "Valid")
        q = zone_bins(quar_f, "CPI", "Quarantine")
        st.altair_chart(grouped_zone_chart(v, q, "Vendas vs CPI (zonas)"), use_container_width=True)


# =============================================================================
# 15) CHECKS (Advanced) - audit / equipa de dados
# =============================================================================
with tab_checks:
    st.subheader("🧱 Checks (Advanced)")
    st.caption("Secção técnica (audit / equipa de dados).")

    # Métricas rápidas do catálogo de checks
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de regras", len(checks_df))

    # Proteções: só calcula se existir col criticality
    c2.metric("Críticas (error)", int((checks_df["criticality"] == "error").sum()) if "criticality" in checks_df.columns else 0)
    c3.metric("Avisos (warn)", int((checks_df["criticality"] == "warn").sum()) if "criticality" in checks_df.columns else 0)

    # Mostra tabela dos checks (sem "detalhe extra" para manter simples)
    cols = [c for c in ["name", "criticality", "check", "filter", "run_config_name"] if c in checks_df.columns]
    st.dataframe(checks_df[cols], use_container_width=True)
