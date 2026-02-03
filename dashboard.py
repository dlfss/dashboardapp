# app.py
# =============================================================================
# DQ Sales Monitor — Versão "3 Segundos" (Ultra Simplificada)
# =============================================================================
# Conceito:
# - Tudo é convertido para "Baixo / Médio / Alto".
# - Cores forçadas (Verde/Vermelho) para leitura imediata.
# - Sem dispersão, sem eixos complexos.

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

# Cores fixas e vibrantes
GREEN = "#00E676"  # Verde Valido
RED = "#FF1744"    # Vermelho Quarentena
AMBER = "#FFB300"  # Amarelo Aviso

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
      .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
      .kpi {{
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 15px;
        background: rgba(255,255,255,0.05);
        text-align: center;
      }}
      .kpi .label {{ font-size: 0.9rem; color: #aaa; margin-bottom: 5px; }}
      .kpi .value {{ font-size: 2rem; font-weight: bold; }}
      
      /* Pills para os estados */
      .pill {{ padding: 5px 10px; border-radius: 20px; font-weight: bold; font-size: 0.8rem; }}
      .valid {{ background-color: rgba(0, 230, 118, 0.2); color: {GREEN}; border: 1px solid {GREEN}; }}
      .quar {{ background-color: rgba(255, 23, 68, 0.2); color: {RED}; border: 1px solid {RED}; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# 3) DADOS (MOCK)
# =============================================================================
def _daterange(start: datetime, weeks: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(weeks)]

@st.cache_data(show_spinner=False)
def make_mock_checks() -> pd.DataFrame:
    # Cria apenas uma tabela dummy de checks para a tab final
    rows = [("Temperature_is_null", "error"), ("Fuel_Price_is_null", "error"), ("Sales_Label_invalid", "error")]
    return pd.DataFrame(rows, columns=["name", "criticality"])

@st.cache_data(show_spinner=False)
def make_mock_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # Gerar dados base
    dates = _daterange(datetime(2023, 1, 1), 50)
    data = []
    
    for d in dates:
        # Simula 40 lojas
        for s in range(1, 41):
            sales = rng.normal(20000, 5000)
            temp = rng.normal(20, 10)  # Celsius
            fuel = rng.normal(3.5, 0.5)
            cpi = rng.normal(200, 10)
            unemp = rng.normal(7, 1.5)
            
            # Introduzir falhas aleatórias
            is_quarantine = rng.random() < 0.15 # 15% quarentena
            
            row = {
                "Date": d,
                "Store": s,
                "Weekly_Sales": abs(sales),
                "Temperature": temp,
                "Fuel_Price": fuel,
                "CPI": cpi,
                "Unemployment": unemp,
                "Holiday_Flag": 1 if rng.random() < 0.1 else 0
            }
            
            # Se for quarentena, estraga alguns dados para "justificar" (nos gráficos não importa tanto o valor errado, mas o grupo)
            if is_quarantine:
                row["_set"] = "Quarantine"
                # Opcional: criar erros reais
                if rng.random() < 0.3: row["Temperature"] = 1000 
            else:
                row["_set"] = "Valid"
            
            data.append(row)
            
    df = pd.DataFrame(data)
    valid = df[df["_set"] == "Valid"].copy()
    quar = df[df["_set"] == "Quarantine"].copy()
    
    # Adicionar colunas falsas de erros para a tab de qualidade
    quar["__errors"] = json.dumps({"items": [{"name": "Check_Falhou", "message": "Erro simulado"}]})
    quar["__warnings"] = None
    
    return make_mock_checks(), valid, quar

def load_data(mode):
    # Simplificação: carrega sempre o mock para garantir que funciona neste exemplo
    # Se quiseres Spark, podes descomentar a lógica original
    return make_mock_data()

# =============================================================================
# 4) FUNÇÕES GRÁFICAS (A MÁGICA SIMPLES)
# =============================================================================

def categorize_column(df, col):
    """Transforma qualquer coluna numérica em: Baixo, Médio, Alto"""
    if df.empty or col not in df.columns:
        return df
    
    # Converter para numérico forçado
    s = pd.to_numeric(df[col], errors='coerce')
    
    # Calcular tercis (33% e 66%)
    q33 = s.quantile(0.33)
    q66 = s.quantile(0.66)
    
    def get_label(x):
        if pd.isna(x): return "N/A"
        if x <= q33: return "Baixo"
        elif x <= q66: return "Médio"
        else: return "Alto"
        
    df[f"{col}_Cat"] = s.apply(get_label)
    return df

def simple_bar_chart(valid_df, quar_df, category_col, title, x_label):
    """Gera um gráfico de barras lado a lado (Verde vs Vermelho)"""
    
    # 1. Preparar dados
    v = valid_df.copy()
    q = quar_df.copy()
    
    # Calcular categorias baseadas no conjunto total (para as escalas serem iguais)
    full = pd.concat([v, q])
    if category_col not in ["Holiday_Flag"]: # Se não for flag, categoriza
        full = categorize_column(full, category_col)
        cat_col_final = f"{category_col}_Cat"
        order = ["Baixo", "Médio", "Alto", "N/A"]
    else:
        full[category_col] = full[category_col].map({0: "Não", 1: "Sim"})
        cat_col_final = category_col
        order = ["Não", "Sim"]

    # Agrupar
    grouped = full.groupby([cat_col_final, "_set"], as_index=False)["Weekly_Sales"].mean()
    
    # 2. Criar Gráfico
    chart = alt.Chart(grouped).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X(f"{cat_col_final}:N", title=x_label, sort=order),
        xOffset=alt.XOffset("_set:N"), # Lado a lado
        y=alt.Y("Weekly_Sales:Q", title="Vendas Médias"),
        color=alt.Color("_set:N", 
                        scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                        legend=alt.Legend(title="Estado", orient="bottom")),
        tooltip=[
            alt.Tooltip(f"{cat_col_final}", title=x_label),
            alt.Tooltip("Weekly_Sales", format=",.0f", title="Vendas"),
            alt.Tooltip("_set", title="Estado")
        ]
    ).properties(
        title=title,
        height=300
    )
    
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(grid=False, labelColor="#ddd", titleColor="#ddd")
        .configure_legend(labelColor="#ddd")
        .configure_title(fontSize=16, color="white")
    )

def simple_trend(valid_df, quar_df):
    v = valid_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    v["_set"] = "Valid"
    q = quar_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    q["_set"] = "Quarantine"
    
    df = pd.concat([v, q])
    
    chart = alt.Chart(df).mark_line(strokeWidth=4).encode(
        x=alt.X("Date:T", title="Data"),
        y=alt.Y("Weekly_Sales:Q", title="Vendas Totais"),
        color=alt.Color("_set:N", scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED])),
        strokeDash=alt.StrokeDash("_set:N", legend=None) # Linha sólida vs tracejada opcional
    ).properties(height=300, title="Evolução Temporal")
    
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridOpacity=0.1, labelColor="#ddd")
        .configure_legend(labelColor="#ddd")
    )

# =============================================================================
# 5) APP PRINCIPAL
# =============================================================================

checks_df, valid_df, quarantine_df = load_data("mock")

st.markdown("## 📊 Monitor de Vendas & Qualidade")
st.markdown(
    f"""
    <div style="margin-bottom: 20px;">
        <span class="pill valid">VALID (Dados Bons)</span> 
        <span class="pill quar">QUARANTINE (Dados Suspeitos)</span>
    </div>
    """, unsafe_allow_html=True
)

# --- KPIs TOPO ---
c1, c2, c3, c4 = st.columns(4)
total_recs = len(valid_df) + len(quarantine_df)
pct_quar = (len(quarantine_df) / total_recs * 100) if total_recs else 0

with c1: 
    st.markdown(f"<div class='kpi'><div class='label'>Total Registos</div><div class='value'>{total_recs:,}</div></div>", unsafe_allow_html=True)
with c2: 
    st.markdown(f"<div class='kpi'><div class='label'>% Quarentena</div><div class='value' style='color:{RED}'>{pct_quar:.1f}%</div></div>", unsafe_allow_html=True)
with c3:
    sales_v = valid_df["Weekly_Sales"].sum()
    st.markdown(f"<div class='kpi'><div class='label'>Vendas (Valid)</div><div class='value' style='color:{GREEN}'>${sales_v/1e6:.1f}M</div></div>", unsafe_allow_html=True)
with c4:
    sales_q = quarantine_df["Weekly_Sales"].sum()
    st.markdown(f"<div class='kpi'><div class='label'>Vendas (Quar.)</div><div class='value' style='color:{RED}'>${sales_q/1e6:.1f}M</div></div>", unsafe_allow_html=True)

st.divider()

# --- INSIGHTS VISUAIS (O PEDIDO PRINCIPAL) ---
st.subheader("🧠 Análise Rápida (3 Segundos)")
st.caption("Comparação de Vendas: Dados Bons (Verde) vs. Dados Suspeitos (Vermelho)")

# Linha 1: Temperatura e Combustível
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(simple_bar_chart(valid_df, quarantine_df, "Temperature", "Impacto da Temperatura", "Temperatura"), use_container_width=True)
with col2:
    st.altair_chart(simple_bar_chart(valid_df, quarantine_df, "Fuel_Price", "Impacto do Combustível", "Preço Combustível"), use_container_width=True)

# Linha 2: Desemprego e CPI
col3, col4 = st.columns(2)
with col3:
    st.altair_chart(simple_bar_chart(valid_df, quarantine_df, "Unemployment", "Impacto do Desemprego", "Nível Desemprego"), use_container_width=True)
with col4:
    st.altair_chart(simple_bar_chart(valid_df, quarantine_df, "CPI", "Impacto do CPI (Inflação)", "Índice Preços"), use_container_width=True)

st.divider()

# --- TENDÊNCIA E DETALHE ---
col_t1, col_t2 = st.columns([2, 1])

with col_t1:
    st.altair_chart(simple_trend(valid_df, quarantine_df), use_container_width=True)

with col_t2:
    st.markdown("### 🚨 Lojas Problemáticas")
    if not quarantine_df.empty:
        top_offenders = quarantine_df["Store"].value_counts().head(5).reset_index()
        top_offenders.columns = ["Loja", "Erros"]
        st.dataframe(top_offenders, hide_index=True, use_container_width=True)
    else:
        st.success("Tudo limpo!")

# --- SECÇÃO DE DEBUG (ESCONDIDA EM EXPANDER) ---
with st.expander("Ver Dados em Bruto (Tabelas)"):
    st.write("Amostra Quarentena:", quarantine_df.head())
