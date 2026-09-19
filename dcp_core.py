# -*- coding: utf-8 -*-
"""
dcp_core.py
===========
Motor de cálculo de la función de distribución de pares de Cooper
Dcp(omega, Tc) para superconductores elementales (Al, Hg, Nb, Ta, Pb) a
partir de salidas DFT/DFPT de Quantum ESPRESSO.

Este módulo contiene ÚNICAMENTE lógica numérica (NumPy/SciPy), sin
matplotlib ni Streamlit, para que pueda importarse y cachearse limpiamente
desde la app interactiva (app.py) o desde un script de línea de comandos.

Autora: Laura S. Arboleda — Física Computacional, Universidad del Tolima
"""
import os
import re
import numpy as np
from scipy.special import expit

_trapz = getattr(np, "trapezoid", None) or np.trapz

# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================
RY_MEV = 13605.693122994          # 1 Ry en meV
CM1_MEV = 0.12398419843320026     # 1 cm^-1 en meV
KB_MEV_K = 0.08617333262          # meV/K

# =============================================================================
# CONFIGURACIÓN NUMÉRICA POR DEFECTO
# =============================================================================
N_OMEGA_DEFAULT = 1500      # puntos de la malla en omega
PUNTOS_POR_KT_DEFAULT = 10  # resolución de la malla en energía electrónica

ELEMENTOS = ["Al", "Hg", "Nb", "Ta", "Pb"]
ORDEN_DISPERSION = ["Al", "Hg", "Ta", "Pb", "Nb"]   # orden por Tc

NOMBRES_LARGOS = {
    "Al": "Aluminio", "Hg": "Mercurio", "Nb": "Niobio",
    "Ta": "Tantalio", "Pb": "Plomo",
}

# Parámetros de referencia de cada elemento (Tc experimental, omega_c de la
# tabla del informe, valores esperados para validar el cálculo, lambda de
# literatura para contraste)
PARAMS = {
    "Al": dict(Tc=1.18, wc=39.676, w_eps=0.0795, w_cp=0.3312, Ncp=1.250486e-18, lam_qe=0.4253, lam_lit=0.43),
    "Hg": dict(Tc=4.15, wc=11.342, w_eps=0.5682, w_cp=1.458,  Ncp=1.943206e-07, lam_qe=2.4657, lam_lit=1.60),
    "Nb": dict(Tc=9.25, wc=26.248, w_eps=0.8942, w_cp=1.7966, Ncp=3.025170e-05, lam_qe=1.4109, lam_lit=0.82),
    "Ta": dict(Tc=4.47, wc=20.639, w_eps=0.5377, w_cp=0.9303, Ncp=3.584773e-08, lam_qe=1.133,  lam_lit=0.69),
    "Pb": dict(Tc=7.19, wc=9.871,  w_eps=1.0088, w_cp=1.9774, Ncp=1.586080e-10, lam_qe=1.0169, lam_lit=1.55),
}

# =============================================================================
# LITERATURA: brecha superconductora Delta0 (meV) reportada para muestras bulk
# Cada tupla es (valor_meV, fuente, nota)
# =============================================================================
LIT = {
    "Al": {
        "teorico": [(0.15,  "Marques 2005 / Continenza 2005, SCDFT (Tabla I)", ""),
                    (0.295, "Kawamura 2026, SCDFT, <Delta> (Tabla S2, arXiv:2603.05123)", ""),
                    (0.418, "Mori 2024, Eliashberg anisotrópico, Delta(T=0) (arXiv:2404.11528)", "")],
        "exp":     [(0.18,  "Giaever 1960", "túnel"),
                    (0.179, "Ashcroft-Mermin (citado en Continenza 2005, Tabla I)", "")],
    },
    "Hg": {
        "teorico": [(0.778, "Tresca 2022, SCDFT: 2Delta/kBTc=4.70, Tc_calc=3.84 K", "derivado")],
        "exp":     [(0.82,  "Bermon-Ginsberg 1964 (2Delta/kBTc=4.60)", "túnel"),
                    (0.82,  "Richards-Tinkham 1960 (2Delta/kBTc=4.6, citado en Tresca 2022)", "derivado")],
    },
    "Nb": {
        "teorico": [(1.55,  "Zarea 2022", "gap anisotrópico"),
                    (1.79,  "Marques 2005 / Continenza 2005, SCDFT (Tabla I)", ""),
                    (1.466, "Kawamura 2026, SCDFT, <Delta> (Tabla S2, arXiv:2603.05123)", "")],
        "exp":     [(2.32,  "Bonnet et al. 1967", "túnel; valor notablemente mayor"),
                    (1.55,  "Pronin et al. 1998", "electrodinámica"),
                    (1.55,  "Ashcroft-Mermin (citado en Continenza 2005, Tabla I)", "")],
    },
    "Ta": {
        "teorico": [(0.76,  "Marques 2005 / Continenza 2005, SCDFT (Tabla I)", ""),
                    (0.766, "Kawamura 2026, SCDFT, <Delta> (Tabla S2, arXiv:2603.05123)", "")],
        "exp":     [(0.71,  "Townsend-Sutton 1962", "túnel"),
                    (0.62,  "Ewert et al. 1981", "túnel"),
                    (0.694, "Ashcroft-Mermin (citado en Continenza 2005, Tabla I)", "")],
    },
    "Pb": {
        "teorico": [(1.31,  "Marques 2005 / Continenza 2005, SCDFT (Tabla I)", ""),
                    (1.24,  "Margine-Giustino 2013, Eliashberg anisotrópico, mu*=0.1", ""),
                    (1.081, "Kawamura 2026, SCDFT con SOC, <Delta> (Tabla S2, arXiv:2603.05123)", "")],
        "exp":     [(1.33,  "Townsend-Sutton 1962", "túnel"),
                    (1.33,  "McMillan-Rowell 1969 (citado en Margine-Giustino 2013)", ""),
                    (1.43,  "Khasanov et al. 2021, muSR monocristal: alpha=2.312, Tc=7.2 K", "derivado")],
    },
}

FALTAN = {
    "Al": "1 experimental",
    "Hg": "2 teóricas y 1 experimental",
    "Ta": "1 teórica",
}


# =============================================================================
# LECTURA DE ARCHIVOS DE QUANTUM ESPRESSO
# =============================================================================
def leer_numerico(ruta):
    """Lee un archivo mezcla de texto y números. Devuelve (matriz, líneas_de_texto)."""
    filas, texto = [], []
    with open(ruta, "r", errors="ignore") as f:
        for linea in f:
            s = linea.strip()
            if not s:
                continue
            texto.append(s)
            if s.startswith("#"):
                continue
            try:
                vals = [float(t) for t in s.replace("D", "E").replace("d", "e").split()]
            except ValueError:
                continue
            filas.append(vals)
    if not filas:
        raise ValueError(f"No se encontraron datos numéricos en {ruta}")
    longitudes = [len(r) for r in filas]
    n = max(set(longitudes), key=longitudes.count)
    data = np.array([r for r in filas if len(r) == n], dtype=float)
    return data, texto


def extraer_lambda(texto):
    for s in texto:
        if "lambda" in s.lower():
            m = re.search(r"lambda\s*[=:]?\s*([0-9]*\.?[0-9]+)", s, re.IGNORECASE)
            if m:
                return float(m.group(1))
    return None


def listar_archivos(carpeta):
    out = []
    for raiz, _, archivos in os.walk(carpeta):
        for a in archivos:
            out.append(os.path.join(raiz, a))
    return out


def buscar_archivos(el, carpeta):
    todos = listar_archivos(carpeta)
    a2f, ph, dos = [], None, None
    for r in todos:
        nom = os.path.basename(r).lower()
        e = el.lower()
        if re.fullmatch(rf"{e}a2f\.dos\d+", nom):
            a2f.append(r)
        elif nom == f"{e}ph.dos":
            ph = r
        elif nom == f"{e}dos.dos":
            dos = r
    if not a2f or ph is None or dos is None:
        raise FileNotFoundError(
            f"[{el}] faltan archivos en '{carpeta}': a2F={len(a2f)} ph.dos={ph is not None} "
            f"dos.dos={dos is not None}")
    return a2f, ph, dos


def elegir_a2f(archivos, lam_objetivo):
    """De los XXa2F.dosN elige el que tenga lambda más cercano al de la tabla."""
    mejor, mejor_d, mejor_lam = None, np.inf, None
    for r in archivos:
        _, txt = leer_numerico(r)
        lam = extraer_lambda(txt)
        if lam is None:
            continue
        d = abs(lam - lam_objetivo)
        if d < mejor_d:
            mejor, mejor_d, mejor_lam = r, d, lam
    if mejor is None:
        mejor = sorted(archivos)[-1]
    return mejor, mejor_lam


def leer_a2f(ruta, wc_fijo=None):
    data, txt = leer_numerico(ruta)
    w = data[:, 0] * RY_MEV          # Ry -> meV
    a2f = data[:, 1]                 # a2F total
    pos = np.where(a2f > 0)[0]
    i = pos[-1]
    if i + 1 < len(w) and a2f[i + 1] <= 0:
        w0, w1, a0, a1 = w[i], w[i + 1], a2f[i], a2f[i + 1]
        wc_calc = w0 + a0 * (w1 - w0) / (a0 - a1)
    else:
        w0, w1, a0, a1 = w[i - 1], w[i], a2f[i - 1], a2f[i]
        wc_calc = w1 + a1 * (w1 - w0) / (a0 - a1) if a0 > a1 else w1
    wc = wc_fijo if wc_fijo else wc_calc
    m = (a2f > 0) & (w > 0) & (w < wc)
    w_full = np.concatenate([[0.0], w[m], [wc]])
    a_full = np.concatenate([[0.0], a2f[m], [0.0]])
    return w_full, a_full, wc_calc, extraer_lambda(txt)


def leer_ph(ruta, wc):
    data, _ = leer_numerico(ruta)
    w = data[:, 0] * CM1_MEV
    n = data[:, 1]
    m = (w > 0) & (w < wc)
    return (np.concatenate([[0.0], w[m], [wc]]),
            np.concatenate([[0.0], n[m], [0.0]]))


def leer_dos(ruta, el, ef_manual=None):
    data, txt = leer_numerico(ruta)
    ef = None
    for s in txt:
        m = re.search(r"EFermi\s*=\s*([-+]?[0-9]*\.?[0-9]+)", s, re.IGNORECASE)
        if m:
            ef = float(m.group(1))
            break
    if ef is None:
        ef = ef_manual
    if ef is None:
        raise ValueError(f"[{el}] no se encontró EFermi en {ruta}")
    x = (data[:, 0] - ef) * 1000.0
    return x, data[:, 1], ef


# =============================================================================
# CÁLCULO DE Dcp
# =============================================================================
def calcular_dcp(x_ne, ne, w_a2f, a2f, w_ph, nph, Tc, wc,
                  n_omega=N_OMEGA_DEFAULT, puntos_por_kt=PUNTOS_POR_KT_DEFAULT):
    """
    Dcp(w) = A(w) B(w) W(w), con
        A(w) = int o(e)  v(e-w)  de ,   B(w) = int o(e') v(e'+w) de'
        W(w) = Nph^2 n(n+1) alpha^2(w) = Nph * a2F * n(n+1)
        Dtilde(e)  = o(e)  int v(e-w)  B(w) W(w) dw
        Dtilde(e') = o(e') int v(e'+w) A(w) W(w) dw
    con o(e)=Ne f(e), v(e)=Ne (1-f(e)), energías respecto a EF.
    """
    kT = KB_MEV_K * Tc
    w = np.linspace(0.0, wc, n_omega)
    dw = w[1] - w[0]
    wt = np.full(n_omega, dw)
    wt[0] = wt[-1] = dw / 2

    a2f_w = np.interp(w, w_a2f, a2f)
    nph_w = np.interp(w, w_ph, nph)
    alpha2 = np.where(nph_w > 1e-12 * nph_w.max(), a2f_w / np.where(nph_w == 0, 1, nph_w), 0.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        n_b = np.zeros_like(w)
        n_b[1:] = 1.0 / np.expm1(w[1:] / kT)
    W = nph_w ** 2 * n_b * (n_b + 1.0) * alpha2
    W[~np.isfinite(W)] = 0.0

    n_e = int(np.clip(np.ceil(2 * wc / (kT / puntos_por_kt)), 2001, 30001))
    if n_e % 2 == 0:
        n_e += 1
    x = np.linspace(-wc, wc, n_e)
    ne_x = np.interp(x, x_ne, ne)
    o = ne_x * expit(-x / kT)

    def v(y):
        return np.interp(y, x_ne, ne) * expit(y / kT)

    A = np.empty(n_omega)
    B = np.empty(n_omega)
    for i, wi in enumerate(w):
        A[i] = _trapz(o * v(x - wi), x)
        B[i] = _trapz(o * v(x + wi), x)
    Dcp = A * B * W
    Ncp = _trapz(Dcp, w)

    De = np.zeros_like(x)
    Dep = np.zeros_like(x)
    for i, wi in enumerate(w):
        if W[i] == 0.0:
            continue
        De += wt[i] * v(x - wi) * B[i] * W[i]
        Dep += wt[i] * v(x + wi) * A[i] * W[i]
    De *= o
    Dep *= o

    w_eps = x[np.argmax(De)]
    w_epsp = x[np.argmax(Dep)]
    return dict(w=w, Dcp=Dcp, Ncp=Ncp, x=x, ne_x=ne_x, De=De, Dep=Dep,
                w_eps=w_eps, w_epsp=w_epsp, delta=abs(w_eps - w_epsp),
                w_cp=w[np.argmax(Dcp)], kT=kT)


def procesar_elemento(el, carpeta_datos, n_omega=N_OMEGA_DEFAULT,
                       puntos_por_kt=PUNTOS_POR_KT_DEFAULT, usar_wc_tabla=True,
                       ef_manual=None):
    """Ejecuta el pipeline completo para un elemento y devuelve un diccionario
    con todas las curvas y resultados escalares necesarios para graficar."""
    p = PARAMS[el]
    a2f_files, ph_file, dos_file = buscar_archivos(el, carpeta_datos)
    a2f_file, lam_arch = elegir_a2f(a2f_files, p["lam_qe"])
    w_a2f, a2f, wc_calc, _ = leer_a2f(a2f_file, wc_fijo=p["wc"] if usar_wc_tabla else None)
    wc = p["wc"] if usar_wc_tabla else wc_calc
    w_ph, nph = leer_ph(ph_file, wc)
    x_ne, ne, ef = leer_dos(dos_file, el, ef_manual=ef_manual)
    cobertura_ok = not (x_ne.min() > -2 * wc or x_ne.max() < 2 * wc)
    calc = calcular_dcp(x_ne, ne, w_a2f, a2f, w_ph, nph, p["Tc"], wc,
                         n_omega=n_omega, puntos_por_kt=puntos_por_kt)
    return dict(
        elemento=el, Tc=p["Tc"], wc=wc, wc_calc=wc_calc,
        w_a2f=w_a2f, a2f=a2f, w_ph=w_ph, nph=nph, calc=calc,
        lam_qe=lam_arch if lam_arch is not None else p["lam_qe"], lam_lit=p["lam_lit"],
        EF=ef, cobertura_ok=cobertura_ok,
        archivo_a2f=os.path.basename(a2f_file),
        archivo_ph=os.path.basename(ph_file),
        archivo_dos=os.path.basename(dos_file),
    )


def procesar_todos(carpeta_datos, elementos=None, **kwargs):
    elementos = elementos or ELEMENTOS
    res = {}
    errores = {}
    for el in elementos:
        try:
            res[el] = procesar_elemento(el, carpeta_datos, **kwargs)
        except (FileNotFoundError, ValueError) as err:
            errores[el] = str(err)
    return res, errores


# =============================================================================
# COMPARACIONES / AJUSTES GLOBALES
# =============================================================================
def medias_lit(el):
    t = [v for v, _, _ in LIT[el]["teorico"]]
    e = [v for v, _, _ in LIT[el]["exp"]]
    return float(np.mean(t)), float(np.mean(e))


def ajuste_lineal(res, elementos=None, excluir=None):
    """Ajuste y = m x + b entre 1/2 Delta_Dcp (x) y la media experimental (y)."""
    elementos = elementos or [e for e in ORDEN_DISPERSION if e in res]
    excluir = excluir or []
    els = [e for e in elementos if e not in excluir]
    x = np.array([res[e]["calc"]["delta"] / 2 for e in els])
    y = np.array([medias_lit(e)[1] for e in els])
    if len(els) < 2:
        return None
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(m=m, b=b, r2=r2, x=x, y=y, elementos=els)


def tabla_resultados_df(res):
    """Devuelve una lista de dicts lista para pandas.DataFrame con los
    resultados numéricos de todos los elementos procesados."""
    filas = []
    for el in ELEMENTOS:
        if el not in res:
            continue
        r = res[el]
        c = r["calc"]
        mt, me = medias_lit(el)
        half = c["delta"] / 2
        filas.append({
            "Elemento": el,
            "Tc (K)": r["Tc"],
            "omega_c (meV)": round(r["wc"], 3),
            "omega_eps (meV)": round(c["w_eps"], 4),
            "omega_eps' (meV)": round(c["w_epsp"], 4),
            "Delta_Dcp (meV)": round(c["delta"], 4),
            "1/2 Delta_Dcp (meV)": round(half, 4),
            "omega_cp (meV)": round(c["w_cp"], 4),
            "Ncp (u.a.)": c["Ncp"],
            "Delta0 teo. media (meV)": round(mt, 3),
            "Delta0 exp. media (meV)": round(me, 3),
            "Error vs exp. (%)": round(100 * (half - me) / me, 1),
            "lambda (DFT/DFPT)": round(r["lam_qe"], 4),
            "lambda (literatura)": r["lam_lit"],
        })
    return filas


def tabla_literatura_df():
    filas = []
    for el in ELEMENTOS:
        for tipo, etiqueta in (("teorico", "Teórico"), ("exp", "Experimental")):
            for v, fuente, nota in LIT[el][tipo]:
                filas.append({
                    "Elemento": el, "Tipo": etiqueta, "Delta0 (meV)": v,
                    "Fuente": fuente, "Nota": nota,
                })
    return filas
