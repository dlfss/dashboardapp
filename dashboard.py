# app.py
# =============================================================================
# DQ Sales Monitor — Versão Final com Insights em Containers
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
# 2) CSS / ESTILO
# =============================================================================
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; }}
      
      .kpi {{
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 15px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      }}
      .kpi .label {{ font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 4px; }}
      .kpi .value {{ font-size: 1.8rem; font-weight: 800; line-height: 1.1; }}
      .kpi .delta {{ font-size: 0.9rem; margin-top: 5px; opacity: 0.8; }}
      
      .pill {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; margin-right: 5px; }}
      .pill-green {{ background: rgba(0,230,118,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
      .pill-red   {{ background: rgba(255,23,68,0.15);  color: {RED};   border: 1px solid {RED}; }}
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 3) DADOS MOCK (Simulação)
# =============================================================================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]

@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    rows = [
        ("Temperature_is_null", "warn", {"function": "is_not_null"}),
        ("Fuel_Price_is_null", "error", {"function": "is_not_null"}),
        ("Sales_Label_invalid", "error", {"function": "is_in_list"}),
        ("Unemployment_range", "error", {"function": "is_in_range"}),
    ]
    return pd.DataFrame(rows, columns=["name", "criticality", "check"])

@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stores = list(range(1, 46))
    start = datetime(2023, 1, 1)
    dates = _daterange(start, 50) 

    rows = []
    id_counter = 1

    for d in dates:
        active_stores = rng.choice(stores, size=rng.integers(30, 46), replace=False)
        for s in active_stores:
            base = rng.normal(1_500_000, 200_000)
            season = 1.0 + 0.15 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi)
            sales = max(0, base * season + rng.normal(0, 50_000))

            rows.append({
                "id": id_counter,
                "Store": int(s),
                "Date": d.date(),
                "Weekly_Sales": float(round(sales, 2)),
                "Temperature": float(np.clip(rng.normal(55, 15), -10, 95)),
                "Fuel_Price": float(np.clip(rng.normal(3.5, 0.4), 2.0, 5.0)),
                "CPI": float(np.clip(rng.normal(215, 8), 190, 240)),
                "Unemployment": float(np.clip(rng.normal(7.0, 1.2), 4.0, 11.0)),
                "Holiday_Flag": int(rng.random() < 0.1)
            })
            id_counter += 1

    df = pd.DataFrame(rows)
    
    quarantine_idx = rng.choice(df.index, size=int(len(df) * 0.15), replace=False)
    q = df.loc[quarantine_idx].copy()
    v = df.drop(quarantine_idx).copy()

    q["__errors"] = json.dumps({"items": [{"name": "Mock_Error", "message": "Simulação de erro"}]})
    q["__warnings"] = None
    v["__errors"] = None
    v["__warnings"] = None

    return v.reset_index(drop=True), q.reset_index(drop=True)

def load_data(mode: str):
    checks = make_mock_checks()
    valid_df, quarantine_df = make_mock_valid_and_quarantine()
    valid_df["Date"] = pd.to_datetime(valid_df["Date"])
    quarantine_df["Date"] = pd.to_datetime(quarantine_df["Date"])
    return checks, valid_df, quarantine_df


# =============================================================================
# 4) FUNÇÕES AUXILIARES
# =============================================================================
def apply_filters(df, start, end, stores):
    mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
    if stores:
        mask &= df["Store"].isin(stores)
    return df[mask]

def kpi_card(label, value, delta=None, color=None):
    c_style = f"color: {color};" if color else ""
    delta_html = f"<div class='delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""<div class='kpi'>
            <div class='label'>{label}</div>
            <div class='value' style='{c_style}'>{value}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True
    )

def base_altair(chart):
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(grid=False, labelColor="#ddd", titleColor="#ccc")
        .configure_legend(labelColor="#ddd")
    )


# =============================================================================
# 5) APP PRINCIPAL
# =============================================================================
st.title("📊 DQ Sales Monitor")

# -- SIDEBAR --
st.sidebar.header("Filtros")
data_mode = st.sidebar.radio("Fonte", ["mock", "databricks"], index=0)
checks_df, valid_df, quarantine_df = load_data(data_mode)

all_dates = pd.concat([valid_df["Date"], quarantine_df["Date"]])
min_d, max_d = all_dates.min().date(), all_dates.max().date()
dates = st.sidebar.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_date, end_date = dates if isinstance(dates, tuple) and len(dates) == 2 else (min_d, max_d)

all_stores = sorted(pd.concat([valid_df["Store"], quarantine_df["Store"]]).unique())
sel_stores = st.sidebar.multiselect("Lojas", all_stores)

valid_f = apply_filters(valid_df, start_date, end_date, sel_stores)
quar_f = apply_filters(quarantine_df, start_date, end_date, sel_stores)

# -- TABS --
tab_over, tab_sales, tab_qual, tab_ins, tab_check = st.tabs(
    ["⚡ Overview", "📈 Vendas", "✅ Qualidade", "🧠 Insights", "🧱 Checks"]
)

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW (Inalterada)
# -----------------------------------------------------------------------------
with tab_over:
    n_val, n_quar = len(valid_f), len(quar_f)
    total = n_val + n_quar
    pct_q = (n_quar / total * 100) if total else 0
    sales_v = valid_f["Weekly_Sales"].sum()
    sales_q = quar_f["Weekly_Sales"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Registos", f"{total:,}")
    with c2: kpi_card("% Quarentena", f"{pct_q:.1f}%", color=RED)
    with c3: kpi_card("Vendas (Valid)", f"${sales_v/1e6:.1f}M", color=GREEN)
    with c4: kpi_card("Vendas (Quar.)", f"${sales_q/1e6:.1f}M", color=RED)

    st.markdown("---")
    
    t_df = pd.concat([valid_f.assign(Tipo="Valid"), quar_f.assign(Tipo="Quarantine")])
    if not t_df.empty:
        trend = t_df.groupby(["Date", "Tipo"], as_index=False)["Weekly_Sales"].sum()
        c = alt.Chart(trend).mark_line(strokeWidth=3).encode(
            x=alt.X("Date", title="Data"),
            y=alt.Y("Weekly_Sales", title="Vendas"),
            color=alt.Color("Tipo", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]))
        ).properties(height=300, title="Evolução Temporal")
        st.altair_chart(base_altair(c), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: VENDAS (Inalterada)
# -----------------------------------------------------------------------------
with tab_sales:
    st.subheader("Performance de Vendas")
    if not valid_f.empty:
        top = valid_f.groupby("Store", as_index=False)["Weekly_Sales"].sum().nlargest(10, "Weekly_Sales")
        c = alt.Chart(top).mark_bar().encode(
            x=alt.X("Weekly_Sales", title="Total Vendas"),
            y=alt.Y("Store:O", sort="-x", title="Loja"),
            color=alt.value(GREEN),
            tooltip=["Store", alt.Tooltip("Weekly_Sales", format=",.0f")]
        ).properties(title="Top 10 Lojas (Dados Válidos)")
        st.altair_chart(base_altair(c), use_container_width=True)
    else:
        st.warning("Sem dados válidos para mostrar.")

# -----------------------------------------------------------------------------
# TAB 3: QUALIDADE (Inalterada)
# -----------------------------------------------------------------------------
with tab_qual:
    st.subheader("Análise de Erros")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div style="padding: 20px; background: rgba(255,23,68,0.1); border-radius: 10px; border: 1px solid {RED};">
                <h3 style="color: {RED}; margin:0;">{n_quar:,}</h3>
                <p style="margin:0;">Registos em quarentena</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        if not quar_f.empty:
            st.dataframe(quar_f[["Date", "Store", "Weekly_Sales", "__errors"]].head(100), use_container_width=True, height=300)
        else:
            st.success("Nenhum registo em quarentena!")

# -----------------------------------------------------------------------------
# TAB 4: INSIGHTS (ALTERADA COM O TEU ESTILO E LOGICA)
# -----------------------------------------------------------------------------
with tab_ins:
    st.markdown("### 🧠 Insights de Dados")
    
    if valid_f.empty and quar_f.empty:
        st.warning("Sem dados.")
    else:
        # Função genérica para criar o gráfico e colocar dentro do container
        def render_insight_container(v_df, q_df, col, title, is_categorical=False, step=1.0):
            with st.container(border=True):
                st.markdown(f"**{title}**")
                
                # 1. Preparar dados
                v = v_df[[col, "Weekly_Sales"]].copy(); v["Status"] = "Valid"
                q = q_df[[col, "Weekly_Sales"]].copy(); q["Status"] = "Quarantine"
                df = pd.concat([v, q], ignore_index=True)
                
                # 2. Criar Eixo X
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=[col])

                sort_order = None
                if is_categorical:
                    # Lógica de Zonas para Temperatura (Baixa/Média/Alta)
                    q33, q66 = df[col].quantile([0.33, 0.66])
                    def get_zone(x): return "Baixa" if x <= q33 else ("Média" if x <= q66 else "Alta")
                    df["Eixo_X"] = df[col].apply(get_zone)
                    sort_order = ["Baixa", "Média", "Alta"]
                else:
                    # Lógica Numérica para os outros (Bins)
                    df["Eixo_X"] = (np.floor(df[col] / step) * step).round(2)
                
                # 3. Agrupar
                grouped = df.groupby(["Eixo_X", "Status"], as_index=False)["Weekly_Sales"].mean()
                
                # 4. Gráfico
                chart = alt.Chart(grouped).mark_bar(cornerRadius=4).encode(
                    x=alt.X("Eixo_X", title=None, sort=sort_order),
                    xOffset="Status",
                    y=alt.Y("Weekly_Sales", title=None, axis=None),
                    color=alt.Color("Status", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]), legend=None),
                    tooltip=["Eixo_X", "Status", alt.Tooltip("Weekly_Sales", format=",.0f")]
                ).properties(height=200) # Altura controlada dentro do container

                st.altair_chart(base_altair(chart), use_container_width=True)

        # Layout Grelha
        c1, c2 = st.columns(2)
        
        with c1:
            # Temperatura: Categórica (Baixa/Média/Alta)
            render_insight_container(valid_f, quar_f, "Temperature", "🌡️ Temperatura (Zonas)", is_categorical=True)
            
        with c2:
            # Combustível: Numérico (Passo 0.5)
            render_insight_container(valid_f, quar_f, "Fuel_Price", "⛽ Combustível (Intervalos $0.5)", is_categorical=False, step=0.5)
            
        c3, c4 = st.columns(2)
        
        with c3:
            # Desemprego: Numérico (Passo 1.0)
            render_insight_container(valid_f, quar_f, "Unemployment", "💼 Desemprego (Intervalos 1%)", is_categorical=False, step=1.0)
            
        with c4:
            # CPI: Numérico (Passo 10.0)
            render_insight_container(valid_f, quar_f, "CPI", "💰 CPI (Intervalos 10)", is_categorical=False, step=10.0)
            
        # Legenda simples
        st.markdown(
            f"""<div style="text-align:center; opacity:0.6; font-size:0.8rem; margin-top:5px;">
                <span style="color:{GREEN}">■</span> Valid &nbsp;&nbsp;&nbsp; 
                <span style="color:{RED}">■</span> Quarantine
            </div>""", unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# TAB 5: CHECKS (Inalterada)
# -----------------------------------------------------------------------------
with tab_check:
    st.subheader("Definição de Regras")
    st.dataframe(checks_df, use_container_width=True)
