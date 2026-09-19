# -*- coding: utf-8 -*-
"""
App interactiva de Streamlit: función de distribución de pares de Cooper
Dcp(omega, Tc) en Al, Hg, Nb, Ta y Pb a partir de datos DFT/DFPT.

Autora: Laura S. Arboleda — Física Computacional, Universidad del Tolima
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

import dcp_core as core

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dcp — Superconductividad convencional",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datos")

# Paleta (coherente con el informe)
C_CURVA = "#5f3dc4"
C_PICO = "#e03131"
C_EPS = "#d6336c"
C_EPSP = "#1098ad"
C_FIT = "#495057"
C_TEO = "#7048e8"
C_EXP = "#0ca678"
C_NUESTRO = "#e03131"


# =============================================================================
# CÁLCULO (CACHEADO)
# =============================================================================
@st.cache_data(show_spinner=False)
def cargar_resultados(n_omega, puntos_por_kt, usar_wc_tabla):
    return core.procesar_todos(
        DATA_DIR,
        n_omega=n_omega,
        puntos_por_kt=puntos_por_kt,
        usar_wc_tabla=usar_wc_tabla,
    )


def picos(x, y, n_max=4, prom_rel=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx, _ = find_peaks(y, prominence=prom_rel * np.nanmax(y))
    if len(idx) == 0:
        idx = np.array([int(np.argmax(y))])
    idx = np.sort(idx[np.argsort(y[idx])[::-1]][:n_max])
    return idx


# =============================================================================
# FIGURAS PLOTLY
# =============================================================================
def fig_linea_con_picos(x, y, titulo, xlab, ylab, n_max=5, prom_rel=0.05, fmt="{:.3f}"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C_CURVA, width=2),
                              name=titulo, hovertemplate=f"{xlab}=%{{x:.3f}}<br>{ylab}=%{{y:.4g}}<extra></extra>"))
    idx = picos(x, y, n_max=n_max, prom_rel=prom_rel)
    fig.add_trace(go.Scatter(
        x=np.asarray(x)[idx], y=np.asarray(y)[idx], mode="markers+text",
        marker=dict(color=C_PICO, size=8, line=dict(color="white", width=1)),
        text=[fmt.format(v) for v in np.asarray(x)[idx]], textposition="top center",
        textfont=dict(color=C_PICO, size=11), showlegend=False,
        hovertemplate=f"pico {xlab}=%{{x:.3f}}<extra></extra>"))
    fig.update_layout(
        title=titulo, xaxis_title=xlab, yaxis_title=ylab,
        margin=dict(l=10, r=10, t=40, b=10), height=340,
        template="plotly_white",
    )
    return fig


def fig_ne(c):
    x, y = c["x"], c["ne_x"]
    ne_ef = float(np.interp(0.0, x, y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C_CURVA, width=2), name="N_e"))
    fig.add_vline(x=0, line=dict(color=C_CURVA, dash="dash", width=1))
    fig.add_trace(go.Scatter(x=[0], y=[ne_ef], mode="markers",
                              marker=dict(color=C_PICO, size=9, line=dict(color="white", width=1)),
                              name=f"N_e(E_F) = {ne_ef:.4f}"))
    fig.update_layout(
        title="Densidad de estados electrónica N_e(ε)", xaxis_title="ε − E_F (meV)", yaxis_title="N_e",
        margin=dict(l=10, r=10, t=40, b=10), height=340, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def fig_dcp(c):
    fig = fig_linea_con_picos(c["w"], c["Dcp"], "D_cp(ω, T_c)", "ω (meV)", "D_cp (u.a.)",
                               n_max=3, prom_rel=0.08)
    fig.add_vline(x=c["w_cp"], line=dict(color=C_PICO, dash="dash", width=1.5),
                  annotation_text=f"ω_cp = {c['w_cp']:.3f} meV", annotation_position="top right")
    return fig


def fig_dtilde(c):
    De = c["De"] / c["De"].max()
    Dep = c["Dep"] / c["Dep"].max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["x"], y=Dep, mode="lines", line=dict(color=C_EPSP, width=2),
                              name=f"D̃_cp(ε'), pico en {c['w_epsp']:+.3f} meV"))
    fig.add_trace(go.Scatter(x=c["x"], y=De, mode="lines", line=dict(color=C_EPS, width=2),
                              name=f"D̃_cp(ε), pico en {c['w_eps']:+.3f} meV"))
    for xp, col in ((c["w_eps"], C_EPS), (c["w_epsp"], C_EPSP)):
        fig.add_vline(x=xp, line=dict(color=col, dash="dash", width=1))
    fig.add_trace(go.Scatter(
        x=[c["w_epsp"], c["w_eps"]], y=[0.5, 0.5], mode="lines+markers",
        line=dict(color=C_FIT, width=1.5), marker=dict(symbol="arrow", size=8, angleref="previous"),
        showlegend=False, hoverinfo="skip"))
    fig.add_annotation(x=(c["w_eps"] + c["w_epsp"]) / 2, y=0.56,
                        text=f"Δ_Dcp = {c['delta']:.3f} meV", showarrow=False,
                        font=dict(color=C_FIT, size=12), bgcolor="white")
    fig.update_layout(
        title="Distribuciones electrónicas D̃_cp(ε) y D̃_cp(ε')",
        xaxis_title="ε − E_F (meV)", yaxis_title="Normalizada (máx. = 1)",
        margin=dict(l=10, r=10, t=40, b=10), height=380, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(range=[0, 1.2]),
    )
    return fig


def fig_fit(res, ajuste):
    fig = go.Figure()
    xs, ys, els = ajuste["x"], ajuste["y"], ajuste["elementos"]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=els, textposition="top center",
        marker=dict(color=C_NUESTRO, size=13, symbol="square",
                    line=dict(color="black", width=1)),
        textfont=dict(size=13, color="#212529"), name="Elementos"))
    xx = np.linspace(0, xs.max() * 1.15, 60)
    signo = "+" if ajuste["b"] >= 0 else "−"
    fig.add_trace(go.Scatter(
        x=xx, y=ajuste["m"] * xx + ajuste["b"], mode="lines",
        line=dict(color=C_FIT, dash="dash", width=2),
        name=f"y = {ajuste['m']:.3f}x {signo} {abs(ajuste['b']):.3f}  (R² = {ajuste['r2']:.2f})"))
    fig.update_layout(
        title="Brecha bibliográfica Δ(0) vs ½Δ_Dcp calculado",
        xaxis_title="½Δ_Dcp (meV)", yaxis_title="Δ(0) bibliográfica, media experimental (meV)",
        margin=dict(l=10, r=10, t=40, b=10), height=430, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def fig_dispersion(res):
    rng = np.random.default_rng(3)
    els = [e for e in core.ORDEN_DISPERSION if e in res]
    fig = go.Figure()
    for i, e in enumerate(els):
        vt = [v for v, _, _ in core.LIT[e]["teorico"]]
        ve = [v for v, _, _ in core.LIT[e]["exp"]]
        fig.add_trace(go.Scatter(
            x=[i - 0.15 + rng.uniform(-0.05, 0.05) for _ in vt], y=vt, mode="markers",
            marker=dict(color=C_TEO, symbol="square", size=10, line=dict(color="black", width=0.5)),
            name="Teórico (lit.)", legendgroup="teo", showlegend=(i == 0),
            hovertemplate=f"{e} teórico: %{{y:.3f}} meV<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[i + 0.15 + rng.uniform(-0.05, 0.05) for _ in ve], y=ve, mode="markers",
            marker=dict(color=C_EXP, symbol="triangle-up", size=11, line=dict(color="black", width=0.5)),
            name="Experimental (lit.)", legendgroup="exp", showlegend=(i == 0),
            hovertemplate=f"{e} experimental: %{{y:.3f}} meV<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[i], y=[res[e]["calc"]["delta"] / 2], mode="markers",
            marker=dict(color=C_NUESTRO, symbol="star", size=20, line=dict(color="black", width=1)),
            name="Nuestro cálculo (½Δ_Dcp)", legendgroup="calc", showlegend=(i == 0),
            hovertemplate=f"{e} calculado: %{{y:.3f}} meV<extra></extra>"))
    fig.update_layout(
        title="Brecha superconductora Δ(0): teoría, experimento y cálculo",
        xaxis=dict(tickmode="array", tickvals=list(range(len(els))), ticktext=els, title="Elemento"),
        yaxis_title="Brecha superconductora (meV)",
        margin=dict(l=10, r=10, t=40, b=10), height=460, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# =============================================================================
# BARRA LATERAL
# =============================================================================
st.sidebar.title("🧊 Dcp — Panel de control")
st.sidebar.markdown(
    "Función de distribución de pares de Cooper $D_{cp}(\\omega,T_c)$ "
    "a partir de cálculos DFT/DFPT (Quantum ESPRESSO)."
)

with st.sidebar.expander("Parámetros numéricos", expanded=False):
    n_omega = st.slider("Puntos de malla en ω", 300, 3000, core.N_OMEGA_DEFAULT, step=100)
    puntos_por_kt = st.slider("Puntos por k_B·T_c (malla electrónica)", 4, 20, core.PUNTOS_POR_KT_DEFAULT)
    usar_wc_tabla = st.checkbox("Usar ω_c de la tabla de referencia", value=True,
                                 help="Si se desactiva, ω_c se calcula extrapolando α²F(ω) a cero.")

with st.spinner("Calculando D_cp para los cinco elementos…"):
    res, errores = cargar_resultados(n_omega, puntos_por_kt, usar_wc_tabla)

if errores:
    for el, msg in errores.items():
        st.sidebar.error(f"{el}: {msg}")

if not res:
    st.error("No se pudo procesar ningún elemento. Verifica que la carpeta `Datos/` "
             "esté junto a `app.py` y contenga los archivos de Quantum ESPRESSO.")
    st.stop()

elementos_disponibles = [e for e in core.ELEMENTOS if e in res]
elegido = st.sidebar.selectbox(
    "Elemento a analizar",
    elementos_disponibles,
    format_func=lambda e: f"{e} — {core.NOMBRES_LARGOS.get(e, '')}",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Datos de partida: salidas DFT/DFPT en formato Quantum ESPRESSO, "
    "provistas por el profesor Guillermo González."
)

# =============================================================================
# ENCABEZADO
# =============================================================================
st.title("Superconductividad convencional: función de distribución de pares de Cooper")
st.markdown(
    "Cálculo de $D_{cp}(\\omega,T_c)$ para los superconductores elementales "
    "**Al, Hg, Nb, Ta y Pb**, a partir de la función de Eliashberg "
    "$\\alpha^2F(\\omega)$, la densidad de estados fonónica $N_{ph}(\\omega)$ y la "
    "densidad de estados electrónica $N_e(\\epsilon)$ obtenidas con DFT/DFPT."
)

tab_elem, tab_comp, tab_lit, tab_metodo = st.tabs(
    ["📊 Análisis por elemento", "📈 Comparación global", "📚 Literatura", "🧮 Metodología"]
)

# =============================================================================
# PESTAÑA 1: ANÁLISIS POR ELEMENTO
# =============================================================================
with tab_elem:
    r = res[elegido]
    c = r["calc"]

    st.subheader(f"{elegido} — {core.NOMBRES_LARGOS.get(elegido, '')}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("T_c experimental", f"{r['Tc']} K")
    col2.metric("ω_c (corte)", f"{r['wc']:.3f} meV")
    col3.metric("ω_cp (máx. D_cp)", f"{c['w_cp']:.3f} meV")
    col4.metric("½Δ_Dcp", f"{c['delta']/2:.3f} meV")
    col5.metric("λ (DFT/DFPT)", f"{r['lam_qe']:.3f}", delta=f"lit. {r['lam_lit']}")

    mt, me = core.medias_lit(elegido)
    if me:
        error_pct = 100 * (c["delta"] / 2 - me) / me
        st.caption(
            f"½Δ_Dcp = {c['delta']/2:.3f} meV frente a Δ₀ media experimental = {me:.3f} meV "
            f"(error {error_pct:+.1f} %) y Δ₀ media teórica = {mt:.3f} meV. "
            f"N_cp = {c['Ncp']:.3e} u.a."
        )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_linea_con_picos(r["w_a2f"], r["a2f"], "α²F(ω) — función de Eliashberg",
                                             "ω (meV)", "α²F", n_max=5), use_container_width=True)
    with g2:
        st.plotly_chart(fig_linea_con_picos(r["w_ph"], r["nph"], "N_ph(ω) — DOS fonónica",
                                             "ω (meV)", "N_ph", n_max=5), use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(fig_ne(c), use_container_width=True)
    with g4:
        st.plotly_chart(fig_dcp(c), use_container_width=True)

    st.plotly_chart(fig_dtilde(c), use_container_width=True)

    with st.expander("Detalles del archivo y de la malla numérica"):
        st.write(
            f"- Archivo α²F: `{r['archivo_a2f']}`  \n"
            f"- Archivo N_ph: `{r['archivo_ph']}`  \n"
            f"- Archivo N_e: `{r['archivo_dos']}`  \n"
            f"- E_F leída del archivo: {r['EF']:.4f} eV  \n"
            f"- ω_c extrapolado de α²F → 0: {r['wc_calc']:.3f} meV "
            f"(usado: {r['wc']:.3f} meV)  \n"
            f"- Cobertura de N_e en ±2ω_c: {'✅ suficiente' if r['cobertura_ok'] else '⚠️ se extendió con el valor del borde'}  \n"
            f"- k_B·T_c = {c['kT']:.4f} meV"
        )

    df_el = pd.DataFrame({
        "ω (meV)": c["w"],
        "D_cp (u.a.)": c["Dcp"],
    })
    st.download_button(
        f"⬇️ Descargar D_cp({elegido}) en CSV",
        df_el.to_csv(index=False).encode("utf-8"),
        file_name=f"{elegido}_Dcp.csv", mime="text/csv",
    )

# =============================================================================
# PESTAÑA 2: COMPARACIÓN GLOBAL
# =============================================================================
with tab_comp:
    st.subheader("Tabla de resultados")
    df_res = pd.DataFrame(core.tabla_resultados_df(res))
    st.dataframe(df_res, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar tabla de resultados (CSV)",
        df_res.to_csv(index=False).encode("utf-8"),
        file_name="tabla_resultados.csv", mime="text/csv",
    )

    st.subheader("Ajuste lineal ½Δ_Dcp vs. brecha experimental")
    excluir_nb = st.checkbox("Excluir Nb del ajuste (como en el informe)", value=False)
    ajuste = core.ajuste_lineal(res, excluir=["Nb"] if excluir_nb else None)
    if ajuste:
        st.plotly_chart(fig_fit(res, ajuste), use_container_width=True)
        st.caption(
            f"Pendiente m = {ajuste['m']:.3f}, intercepto b = {ajuste['b']:.3f} meV, "
            f"R² = {ajuste['r2']:.3f}."
        )

    st.subheader("Dispersión: teoría, experimento y cálculo")
    st.plotly_chart(fig_dispersion(res), use_container_width=True)

    st.subheader("Constante de acoplamiento electrón-fonón λ")
    df_lam = pd.DataFrame([
        {"Elemento": e, "λ (DFT/DFPT)": round(res[e]["lam_qe"], 4), "λ (literatura)": res[e]["lam_lit"]}
        for e in elementos_disponibles
    ])
    st.dataframe(df_lam, use_container_width=True, hide_index=True)

# =============================================================================
# PESTAÑA 3: LITERATURA
# =============================================================================
with tab_lit:
    st.subheader("Valores de referencia de la brecha superconductora Δ₀")
    st.caption(
        "Valores en meV para muestras *bulk* de cada elemento puro. "
        "Los marcados como *derivado* se calcularon a partir de la razón "
        "2Δ₀/k_BT_c reportada en la fuente."
    )
    df_lit = pd.DataFrame(core.tabla_literatura_df())
    filtro_el = st.multiselect("Filtrar por elemento", core.ELEMENTOS, default=core.ELEMENTOS)
    st.dataframe(df_lit[df_lit["Elemento"].isin(filtro_el)], use_container_width=True, hide_index=True)

    if core.FALTAN:
        st.info(
            "Referencias aún incompletas para llegar a 3 teóricas + 3 experimentales: "
            + "; ".join(f"{e}: {t}" for e, t in core.FALTAN.items())
        )

# =============================================================================
# PESTAÑA 4: METODOLOGÍA
# =============================================================================
with tab_metodo:
    st.markdown(
        r"""
### Definición

$$
D_{cp}(\omega,T_c)=\int\!\!\int
g^o_e(\epsilon)\,g^v_e(\epsilon-\omega)\,g^{n+1}_{ph}(\omega)\,
g^o_e(\epsilon')\,g^v_e(\epsilon'+\omega)\,g^{n}_{ph}(\omega)\,
\alpha^2(\omega)\,d\epsilon\,d\epsilon'
$$

Como el integrando factoriza en partes que dependen solo de $\epsilon$ o
solo de $\epsilon'$, la integral doble se reduce a un producto de
integrales simples:

$$
D_{cp}(\omega,T_c) = A(\omega)\,B(\omega)\,W(\omega)
$$

con

$$
A(\omega)=\int_{-\omega_c}^{\omega_c} N_e(x)f(x)\,N_e(x-\omega)[1-f(x-\omega)]\,dx,\qquad
B(\omega)=\int_{-\omega_c}^{\omega_c} N_e(x')f(x')\,N_e(x'+\omega)[1-f(x'+\omega)]\,dx'
$$

$$
W(\omega)=N_{ph}(\omega)\,\alpha^2F(\omega)\,n_B(\omega)[n_B(\omega)+1]
$$

El número de pares de Cooper se estima como
$N_{cp}=\int_0^{\omega_c} D_{cp}(\omega,T_c)\,d\omega$, y la separación
entre los picos de las distribuciones electrónicas
$\tilde D_{cp}(\epsilon)$ y $\tilde D_{cp}(\epsilon')$ define
$\Delta_{D_{cp}}=\omega_\epsilon-\omega_{\epsilon'}$, comparado en este
trabajo con $\tfrac12\Delta_{D_{cp}}$ frente a la brecha $\Delta_0$ de la
literatura.

### Implementación numérica

- Las integrales se evalúan con la **regla del trapecio**.
- Malla en $\omega$: puntos uniformes en $[0,\omega_c]$ (ajustable en la barra lateral).
- Malla en energía electrónica: cubre $[-\omega_c,\omega_c]$ con paso
  proporcional a $k_BT_c$, de modo que resuelve el ancho térmico de las
  funciones de Fermi.
- $\alpha^2F$, $N_{ph}$ y $N_e$ se **interpolan linealmente**; $N_e$ se
  prolonga con el valor del borde donde el archivo no llega.
- Los factores de Fermi se evalúan con la función logística estable
  (`scipy.special.expit`) para evitar desbordamientos numéricos.

### Datos de entrada

Cada elemento requiere tres archivos de Quantum ESPRESSO en la carpeta
`Datos/`:

| Archivo | Contenido | Unidad original |
|---|---|---|
| `{El}a2F.dosN` | Función de Eliashberg α²F(ω) | ω en Ry |
| `{El}ph.dos` | Densidad de estados fonónica N_ph(ω) | ω en cm⁻¹ |
| `{El}dos.dos` | Densidad de estados electrónica N_e(E) | E en eV, con E_F en el encabezado |

Todas las energías se convierten internamente a **meV**.
        """
    )
    st.markdown(
        "Datos de partida brindados por el profesor Guillermo González. "
        "Formalismo de $D_{cp}(\\omega,T_c)$ según sus notas de cátedra "
        "(Universidad del Tolima, 2026)."
    )

# =============================================================================
# PIE DE PÁGINA
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#868e96; font-size:0.9em; padding-bottom:1em;'>"
    "Laura Arboleda — Universidad del Tolima"
    "</div>",
    unsafe_allow_html=True,
)
