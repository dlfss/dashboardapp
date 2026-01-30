import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Estatísticas da Empresa", page_icon="📊", layout="wide")

# =========================
# STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "intro"
if "month_cursor" not in st.session_state:
    st.session_state.month_cursor = datetime(2016, 1, 1)

# chat
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False  # começa fechado 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! 👋 Pergunta-me coisas sobre as métricas (ex.: “resumo do mês”, “total do ano”, “top 5 dias por € ganho”)."}
    ]

# contexto calculado uma vez por rerun
if "ctx" not in st.session_state:
    st.session_state.ctx = {}

# =========================
# DATA SOURCE (FAKE vs WAREHOUSE)  ✅ 
# =========================
# Define no Databricks Apps (Environment Variable):
# DATA_SOURCE=fake   ou   DATA_SOURCE=warehouse
DATA_SOURCE = os.environ.get("DATA_SOURCE", "fake").lower().strip()

# =========================
# CSS
# =========================
st.markdown(
    """
<style>
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }

.hero {
  padding: 3rem 2.5rem;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
}
.hero-sub { opacity: 0.85; font-size: 1.05rem; margin-top: .4rem; }
.hero-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 1.25rem; }
.hero-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 1rem 1.05rem;
  background: rgba(255,255,255,0.03);
}

.kpi-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 1.25rem 1.25rem;
  background: rgba(255,255,255,0.03);
}

.month-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: .65rem .75rem;
  background: rgba(255,255,255,0.04);
  min-width: 260px;
}
.month-title { font-weight: 900; text-align: center; font-size: 1.05rem; }
div[data-testid="stPopover"] button { border-radius: 12px !important; font-weight: 800 !important; }

.chat-shell {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  background: rgba(15, 15, 18, 0.70);
  box-shadow: 0 12px 26px rgba(0,0,0,0.35);
  overflow: hidden;
}
.chat-head {
  padding: 12px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 900;
}
.chat-body { padding: 10px 12px 12px 12px; }
.sticky { position: sticky; top: 18px; }

div[data-testid="stChatInput"] textarea{ border-radius: 12px !important; }
.chat-body [data-testid="stChatMessage"]{ margin-bottom: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
MESES_PT = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]

def month_name_pt(m: int) -> str:
    return MESES_PT[m - 1]

def add_months(dt: datetime, delta_months: int) -> datetime:
    y = dt.year
    m = dt.month + delta_months
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return datetime(y, m, 1)

def fmt_int(x: int) -> str:
    return f"{int(x):,}".replace(",", " ")

def fmt_money(x: float) -> str:
    return f"€ {x:,.2f}".replace(",", " ").replace(".", ",")

def fmt_km(x: float) -> str:
    return f"{x:,.1f} km".replace(",", " ").replace(".", ",")

def summarize_df(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"days": 0, "trips_total": 0, "revenue_total": 0.0, "km_total": 0.0,
                "trips_avg": 0.0, "revenue_avg": 0.0, "km_avg": 0.0}
    return {
        "days": int(len(df)),
        "trips_total": int(df["Viagens"].sum()),
        "revenue_total": float(df["€ ganho"].sum()),
        "km_total": float(df["Km"].sum()),
        "trips_avg": float(df["Viagens"].mean()),
        "revenue_avg": float(df["€ ganho"].mean()),
        "km_avg": float(df["Km"].mean()),
    }

# =========================
# DATA LAYER
# =========================
def _spark_available() -> bool:
    return "spark" in globals()

@st.cache_data(ttl=600, show_spinner=False)
def load_daily_year_fake(year: int) -> pd.DataFrame:
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    rng = np.random.default_rng(year)

    weekday = dates.dayofweek.values
    weekly_factor = np.where(weekday < 5, 1.0, 0.80)

    day_of_year = dates.dayofyear.values
    seasonal = 1.0 + 0.20 * np.sin((day_of_year / 365.0) * 2 * np.pi)

    base_trips = 240 + 70 * seasonal
    noise = rng.normal(0, 24, size=len(dates))
    trips = np.maximum(70, (base_trips * weekly_factor + noise)).round().astype(int)

    avg_fare = np.clip(rng.normal(14.2, 1.0, size=len(dates)), 9.5, 22.0)
    avg_km_trip = np.clip(rng.normal(4.6, 0.9, size=len(dates)), 1.2, 13.0)

    revenue = trips * avg_fare
    km = trips * avg_km_trip

    return pd.DataFrame({"date": dates, "Viagens": trips, "€ ganho": revenue, "Km": km})

@st.cache_data(ttl=600, show_spinner=False)
def load_daily_year_from_warehouse(year: int) -> pd.DataFrame:
    if not _spark_available():
        raise RuntimeError("Spark não disponível. Em Databricks normalmente tens 'spark'.")

    query = f"""
    SELECT
      CAST(trip_date AS DATE) AS date,
      COUNT(*) AS Viagens,
      SUM(fare_eur) AS `€ ganho`,
      SUM(distance_km) AS Km
    FROM db.viagens
    WHERE trip_date >= '{year}-01-01' AND trip_date < '{year+1}-01-01'
    GROUP BY CAST(trip_date AS DATE)
    ORDER BY date
    """
    sdf = spark.sql(query)
    pdf = sdf.toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    return pdf

def daily_metrics_for_year(year: int) -> pd.DataFrame:
    return load_daily_year_from_warehouse(year) if DATA_SOURCE == "warehouse" else load_daily_year_fake(year)

@st.cache_data(ttl=600, show_spinner=False)
def metrics_for_month_fake(dt: datetime):
    df_year = load_daily_year_fake(dt.year)
    df_month = df_year[df_year["date"].dt.month == dt.month]
    trips = int(df_month["Viagens"].sum())
    revenue = float(df_month["€ ganho"].sum())
    km_total = float(df_month["Km"].sum())
    avg_fare = (revenue / trips) if trips else 0.0
    avg_km = (km_total / trips) if trips else 0.0
    return trips, avg_fare, avg_km

@st.cache_data(ttl=600, show_spinner=False)
def metrics_for_month_from_warehouse(dt: datetime):
    if not _spark_available():
        raise RuntimeError("Spark não disponível. Em Databricks normalmente tens 'spark'.")

    start = datetime(dt.year, dt.month, 1)
    end = add_months(start, 1)

    query = f"""
    SELECT
      COUNT(*) AS trips,
      AVG(fare_eur) AS avg_fare,
      AVG(distance_km) AS avg_km
    FROM db.viagens
    WHERE trip_date >= '{start.date()}' AND trip_date < '{end.date()}'
    """
    row = spark.sql(query).toPandas().iloc[0]
    return int(row["trips"]), float(row["avg_fare"]), float(row["avg_km"])

def metrics_for_month(dt: datetime):
    return metrics_for_month_from_warehouse(dt) if DATA_SOURCE == "warehouse" else metrics_for_month_fake(dt)

# =========================
# CONTEXT
# =========================
def compute_context():
    cur = st.session_state.month_cursor
    prev = add_months(cur, -1)

    df_year = daily_metrics_for_year(cur.year)
    df_month = df_year[df_year["date"].dt.month == cur.month].copy()

    df_prev_year = daily_metrics_for_year(prev.year)
    df_prev_month = df_prev_year[df_prev_year["date"].dt.month == prev.month].copy()

    st.session_state.ctx = {
        "cur": cur,
        "prev": prev,
        "df_year": df_year,
        "df_month": df_month,
        "df_prev_month": df_prev_month,
        "sum_year": summarize_df(df_year),
        "sum_month": summarize_df(df_month),
        "sum_prev": summarize_df(df_prev_month),
    }

# =========================
# CHAT
# =========================
def chat_answer(prompt: str) -> str:
    p = (prompt or "").strip().lower()
    ctx = st.session_state.ctx
    cur, prev = ctx["cur"], ctx["prev"]
    df_year = ctx["df_year"]
    sum_year, sum_month, sum_prev = ctx["sum_year"], ctx["sum_month"], ctx["sum_prev"]

    if any(k in p for k in ["ajuda", "help", "exemplos"]):
        return (
            "Podes perguntar, por exemplo:\n"
            "- **melhor dia do ano em viagens**\n"
            "- **pior dia do ano em € ganho**\n"
            "- **top 5 dias por km**\n"
            "- **total do ano** / **média por dia**\n"
            "- **resumo do mês**\n"
            "- **comparar com o mês anterior**"
        )

    metric = None
    if "viagem" in p:
        metric = "Viagens"
    elif "€" in p or "euro" in p or "dinheiro" in p or "ganho" in p or "receita" in p:
        metric = "€ ganho"
    elif "km" in p or "quil" in p or "kil" in p:
        metric = "Km"

    want_top = ("top" in p) or ("melhores" in p) or ("maiores" in p)
    want_worst = ("pior" in p) or ("piores" in p) or ("menor" in p)

    if ("melhor dia" in p) or ("pior dia" in p) or want_top or want_worst:
        if metric is None:
            metric = "Viagens"
        df = df_year.sort_values(metric, ascending=want_worst)
        n = 5 if ("top 5" in p or "5" in p) else 1
        sel = df.head(n)

        def valtxt(v):
            if metric == "Viagens":
                return fmt_int(v)
            if metric == "€ ganho":
                return fmt_money(float(v))
            return fmt_km(float(v))

        if n == 1:
            r = sel.iloc[0]
            label = "pior" if want_worst or ("pior dia" in p) else "melhor"
            return f"O **{label} dia do ano** em **{metric}** foi **{r['date'].date()}** com **{valtxt(r[metric])}**."
        else:
            label = "piores" if want_worst else "melhores"
            lines = [f"- {r['date'].date()}: **{valtxt(r[metric])}**" for _, r in sel.iterrows()]
            return f"Top {n} **{label} dias** do ano em **{metric}**:\n" + "\n".join(lines)

    if "total do ano" in p or ("total" in p and "ano" in p):
        return (
            f"**Total do ano {cur.year}:**\n"
            f"- Viagens: **{fmt_int(sum_year['trips_total'])}**\n"
            f"- € ganho: **{fmt_money(sum_year['revenue_total'])}**\n"
            f"- Km: **{fmt_km(sum_year['km_total'])}**"
        )

    if "média" in p and ("ano" in p or "por dia" in p):
        return (
            f"**Média diária no ano {cur.year}:**\n"
            f"- Viagens/dia: **{sum_year['trips_avg']:.1f}**\n"
            f"- € ganho/dia: **{fmt_money(sum_year['revenue_avg'])}**\n"
            f"- Km/dia: **{fmt_km(sum_year['km_avg'])}**"
        )

    if "resumo do mês" in p or ("resumo" in p and ("mês" in p or "mes" in p)):
        return (
            f"**Resumo de {month_name_pt(cur.month)} {cur.year}:**\n"
            f"- Dias: **{sum_month['days']}**\n"
            f"- Viagens (total): **{fmt_int(sum_month['trips_total'])}**\n"
            f"- € ganho (total): **{fmt_money(sum_month['revenue_total'])}**\n"
            f"- Km (total): **{fmt_km(sum_month['km_total'])}**"
        )

    if "compar" in p or "mês anterior" in p or "mes anterior" in p:
        def pct(currv, prevv):
            if prevv == 0:
                return None
            return (currv - prevv) / prevv * 100.0

        pt = pct(sum_month["trips_total"], sum_prev["trips_total"])
        pr = pct(sum_month["revenue_total"], sum_prev["revenue_total"])
        pk = pct(sum_month["km_total"], sum_prev["km_total"])

        def pct_txt(x):
            if x is None:
                return "—"
            sign = "+" if x >= 0 else ""
            return f"{sign}{x:.1f}%"

        return (
            f"**Comparação: {month_name_pt(cur.month)} {cur.year} vs {month_name_pt(prev.month)} {prev.year}:**\n"
            f"- Viagens: **{fmt_int(sum_month['trips_total'])}** vs **{fmt_int(sum_prev['trips_total'])}**  ({pct_txt(pt)})\n"
            f"- € ganho: **{fmt_money(sum_month['revenue_total'])}** vs **{fmt_money(sum_prev['revenue_total'])}**  ({pct_txt(pr)})\n"
            f"- Km: **{fmt_km(sum_month['km_total'])}** vs **{fmt_km(sum_prev['km_total'])}**  ({pct_txt(pk)})"
        )

    return "Tenta: **ajuda**, **resumo do mês**, **total do ano**, **melhor dia do ano em viagens**."

# =========================
# INTRO
# =========================
def render_intro():
    st.markdown(
        f"""
<div class="hero">
  <h1>📊 Estatísticas da Empresa</h1>
  <div class="hero-sub">Dashboard com métricas mensais, gráficos diários e chat. <span style="opacity:.75">Fonte: <b>{DATA_SOURCE}</b></span></div>
  <div class="hero-cards">
    <div class="hero-card"><h4>🗓️ Mensal</h4><p>Navega mês a mês sem limites.</p></div>
    <div class="hero-card"><h4>📈 Gráficos</h4><p>Viagens/€ ganho/Km por dia.</p></div>
    <div class="hero-card"><h4>💬 Chat</h4><p>Painel à direita .</p></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    center = st.columns([1, 0.7, 1])[1]
    with center:
        if st.button("Continuar →", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

# =========================
# TOP BAR
# =========================
def render_topbar():
    left, right = st.columns([0.68, 0.32], vertical_alignment="center")
    with left:
        a, b = st.columns([0.08, 0.92], vertical_alignment="center")
        with a:
            if st.button("←", key="back_to_intro"):
                st.session_state.page = "intro"
                st.rerun()
        with b:
            st.markdown("## Estatísticas — Visão geral")
            st.caption("Seleciona o mês no topo direito para atualizar as métricas.")

    with right:
        st.markdown('<div class="month-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([0.2, 0.6, 0.2], vertical_alignment="center")
        with c1:
            if st.button("◀", use_container_width=True, key="prev_month"):
                st.session_state.month_cursor = add_months(st.session_state.month_cursor, -1)
                st.rerun()
        with c2:
            st.markdown(f'<div class="month-title">{month_name_pt(st.session_state.month_cursor.month)}</div>', unsafe_allow_html=True)
        with c3:
            if st.button("▶", use_container_width=True, key="next_month"):
                st.session_state.month_cursor = add_months(st.session_state.month_cursor, +1)
                st.rerun()

        years = list(range(2000, 2036))
        current_year = st.session_state.month_cursor.year
        with st.popover(f"{current_year}", use_container_width=True):
            chosen = st.selectbox("Ano", years, index=years.index(current_year), label_visibility="collapsed")
            if chosen != current_year:
                st.session_state.month_cursor = datetime(chosen, st.session_state.month_cursor.month, 1)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# KPIs
# =========================
def render_kpis():
    dt = st.session_state.month_cursor
    trips, avg_fare, avg_km = metrics_for_month(dt)

    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.subheader("📌 Visão geral")
    st.write(f"Métricas para **{month_name_pt(dt.month)} {dt.year}**. Fonte: **{DATA_SOURCE}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Quantidade de viagens", fmt_int(trips))
    with c2:
        st.metric("Tarifa média", fmt_money(avg_fare))
    with c3:
        st.metric("Distância média", fmt_km(avg_km))
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CHARTS
# =========================
def render_daily_charts():
    ctx = st.session_state.ctx
    year = ctx["cur"].year
    df_year = ctx["df_year"].copy()

    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.subheader("📈 Métricas diárias")
    st.caption(f"Ano: **{year}** · Fonte: **{DATA_SOURCE}**")

    metrics = ["Viagens", "€ ganho", "Km"]
    with st.container(border=True):
        chosen_metrics = st.multiselect("Métricas", metrics, default=metrics)
        rolling_average = st.toggle("Rolling average (7 dias)", value=False)
        view = st.selectbox("Vista", ["Ano inteiro", "Mês selecionado"], index=0)

    data = df_year
    if view == "Mês selecionado":
        m = st.session_state.month_cursor.month
        data = data[data["date"].dt.month == m]

    if not chosen_metrics:
        st.info("Seleciona pelo menos 1 métrica.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    plot_df = data.set_index("date")[chosen_metrics]
    if rolling_average:
        plot_df = plot_df.rolling(7).mean().dropna()

    tab1, tab2 = st.tabs(["Chart", "Dataframe"])
    tab1.line_chart(plot_df, height=280)
    tab2.dataframe(plot_df, height=280, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CHAT PANEL
# =========================
def render_chat_panel():
    st.markdown('<div class="sticky">', unsafe_allow_html=True)
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown('<div class="chat-head">💬 Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-body">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.45, 0.28, 0.27], vertical_alignment="center")
    with c1:
        st.caption("Ex.: “resumo do mês”, “total do ano”, “top 5 dias por € ganho”.")
    with c2:
        if st.button("Ajuda", use_container_width=True):
            st.session_state.messages.append({"role": "assistant", "content": chat_answer("ajuda")})
            st.rerun()
    with c3:
        if st.button("Limpar", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Chat limpo ✅ Podes perguntar de novo."}]
            st.rerun()

    if st.session_state.chat_open:
        if st.button("Fechar chat", use_container_width=True):
            st.session_state.chat_open = False
            st.rerun()
    else:
        if st.button("Abrir chat", use_container_width=True):
            st.session_state.chat_open = True
            st.rerun()

    st.write("")

    if st.session_state.chat_open:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Pergunta algo…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            answer = chat_answer(prompt)

            with st.chat_message("assistant"):
                ph = st.empty()
                full = ""
                for chunk in re.split(r"(\s+)", answer):
                    full += chunk
                    time.sleep(0.006)
                    ph.markdown(full + "▌")
                ph.markdown(full)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
    else:
        st.info("Chat fechado. Clica em Abrir chat.")

    st.markdown("</div></div></div>", unsafe_allow_html=True)

# =========================
# DASHBOARD
# =========================
def render_dashboard():
    compute_context()
    render_topbar()
    st.write("")

    left, right = st.columns([0.70, 0.30], gap="large")
    with left:
        render_kpis()
        st.write("")
        render_daily_charts()
    with right:
        render_chat_panel()

# =========================
# ROUTER
# =========================
if st.session_state.page == "intro":
    render_intro()
else:
    render_dashboard()

