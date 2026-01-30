def binned_avg_sales(df: pd.DataFrame, xcol: str, label: str, bins: int = 6) -> pd.DataFrame:
    """Cria bins (quantis) e devolve média de Weekly_Sales por bin. Simples e legível."""
    if len(df) == 0 or xcol not in df.columns or "Weekly_Sales" not in df.columns:
        return pd.DataFrame(columns=["Bin", "AvgSales", "Set"])

    d = df[[xcol, "Weekly_Sales"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d["Weekly_Sales"] = pd.to_numeric(d["Weekly_Sales"], errors="coerce")
    d = d.dropna()

    if len(d) < 50:
        return pd.DataFrame(columns=["Bin", "AvgSales", "Set"])

    try:
        d["Bin"] = pd.qcut(d[xcol], q=bins, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["Bin", "AvgSales", "Set"])

    g = d.groupby("Bin", as_index=False)["Weekly_Sales"].mean()
    g["Bin"] = g["Bin"].astype(str)
    g = g.rename(columns={"Weekly_Sales": "AvgSales"})
    g["Set"] = label
    return g


def line_bins_chart(df: pd.DataFrame, title: str):
    """Linha grossa + pontos, cor por Set (Valid/Quarantine)."""
    if len(df) == 0:
        return None
    return (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=4)
        .encode(
            x=alt.X("Bin:N", title="", sort=None),
            y=alt.Y("AvgSales:Q", title="Média Weekly Sales"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(domain=["Valid", "Quarantine"], range=[GREEN, RED]),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=["Set:N", "Bin:N", alt.Tooltip("AvgSales:Q", format=",.0f")],
        )
        .properties(height=360, title=title)
    )
