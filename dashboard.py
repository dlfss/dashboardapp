# app.py
# DQ Sales Monitor — Visual "3 segundos"

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

GREEN = "#00E676"
RED = "#FF1744"
AMBER = "#FFB300"

alt.themes.enable("dark")


# ----------------------------
# CSS
# ----------------------------
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


# ----------------------------
# Mock data generators
# ----------------------------
def _daterange(start: datetime, weeks: int):
    return [start + timedelta(days=7 * i) for i in range(weeks)]


@st.cache_data(show_spinner=False)
def make_mock_checks():
    rows = [
        ("Sales_Label_is_null","error",{"function":"is_not_null","arguments":{"column":"Sales_Label"}}),
        ("Fuel_Price_is_null","error",{"function":"is_not_null","arguments":{"column":"Fuel_Price"}}),
        ("Temperature_isnt_in_range","error",{"function":"is_in_range","arguments":{"column":"Temperature","min_limit":-10,"max_limit":55}}),
    ]
    df = pd.DataFrame(rows, columns=["name","criticality","check"])
    df["filter"]=None
    df["run_config_name"]="default"
    df["check"]=df["check"].apply(json.dumps)
    return df


def _sales_label(v,q33,q66):
    if v>=q66: return "High"
    if v>=q33: return "Medium"
    return "Low"


@st.cache_data(show_spinner=False)
def make_mock_valid_and_quarantine(seed=42):
    rng=np.random.default_rng(seed)
    stores=list(range(1,46))
    start=datetime(2010,2,5)
    dates=_daterange(start,120)

    rows=[]
    i=1
    for d in dates:
        for s in rng.choice(stores,size=35,replace=False):
            ws=max(0,rng.normal(1_500_000,200_000))
            rows.append(dict(
                id=i,Store=s,Date=d.date(),
                Weekly_Sales=ws,
                Holiday_Flag=int(rng.random()<0.1),
                Temperature=rng.normal(50,15),
                Fuel_Price=rng.normal(2.7,0.3),
                CPI=rng.normal(210,5),
                Unemployment=rng.normal(8,0.5)
            ))
            i+=1

    df=pd.DataFrame(rows)
    q33,q66=df.Weekly_Sales.quantile([0.33,0.66])
    df["Sales_Label"]=df.Weekly_Sales.apply(lambda v:_sales_label(v,q33,q66))

    q=df.sample(frac=0.1,random_state=seed).copy()
    v=df.drop(q.index).copy()

    q["__errors"]=None
    q["__warnings"]=None
    v["__errors"]=None
    v["__warnings"]=None

    return v.reset_index(drop=True),q.reset_index(drop=True)


def load_data(mode):
    checks=make_mock_checks()
    v,q=make_mock_valid_and_quarantine()
    return checks,v,q


# ----------------------------
# Header
# ----------------------------
st.markdown(
    f"""
    # 📊 DQ Sales Monitor
    <span class="pill pill-green">VALID</span>
    <span class="pill pill-red">QUARANTINE</span>
    <span class="pill pill-amber">WARNINGS</span>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Controlo")
checks_df, valid_df, quarantine_df = load_data("mock")

show_chat = st.sidebar.toggle("💬 Chat", value=True)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages=[{"role":"assistant","content":"Faz uma pergunta."}]

if show_chat:
    with st.sidebar.expander("💬 Chat",expanded=True):
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        msg=st.chat_input("Escreve aqui…")
        if msg:
            st.session_state.chat_messages.append({"role":"user","content":msg})
            st.session_state.chat_messages.append({"role":"assistant","content":"Recebido."})


# ----------------------------
# Tabs (SEM Chat IA)
# ----------------------------
tab_overview, tab_sales, tab_quality, tab_insights, tab_checks = st.tabs(
    ["⚡ Overview","📈 Vendas","✅ Qualidade","🧠 Insights","🧱 Checks (Advanced)"]
)


# ----------------------------
# OVERVIEW
# ----------------------------
with tab_overview:
    st.write("Overview aqui")


# ----------------------------
# VENDAS
# ----------------------------
with tab_sales:
    st.write("Vendas aqui")


# ----------------------------
# QUALIDADE
# ----------------------------
with tab_quality:
    st.write("Qualidade aqui")


# ----------------------------
# INSIGHTS
# ----------------------------
with tab_insights:
    st.write("Insights aqui")


# ----------------------------
# CHECKS
# ----------------------------
with tab_checks:
    st.dataframe(checks_df)
