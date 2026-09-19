# Dcp — Superconductividad convencional (Al, Hg, Nb, Ta, Pb)

App interactiva en **Streamlit** para calcular y visualizar la función de
distribución de pares de Cooper $D_{cp}(\omega,T_c)$ de cinco
superconductores elementales (Al, Hg, Nb, Ta, Pb) a partir de datos
DFT/DFPT (Quantum ESPRESSO), y compararla con la brecha superconductora
$\Delta_0$ reportada en la literatura.

Autora: **Laura Arboleda** — Física Computacional, Universidad del Tolima.

## Estructura del proyecto

```
.
├── app.py              # App de Streamlit (interfaz e interactividad)
├── dcp_core.py         # Motor de cálculo (NumPy/SciPy, sin dependencias de UI)
├── Datos/              # Archivos de salida de Quantum ESPRESSO (a2F, ph.dos, dos.dos)
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

`dcp_core.py` contiene toda la física (lectura de archivos, conversión de
unidades, cálculo de $D_{cp}$, ajustes y tablas de literatura) y puede
reutilizarse desde cualquier script sin necesidad de Streamlit. `app.py`
solo se encarga de la interfaz y de las gráficas interactivas (Plotly).

## Ejecutar en local (Visual Studio Code)

1. Clona el repositorio y entra en la carpeta:
   ```bash
   git clone https://github.com/<tu-usuario>/<tu-repo>.git
   cd <tu-repo>
   ```
2. Crea un entorno virtual (opcional pero recomendado) y actívalo:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # en Windows: .venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```
   Se abrirá automáticamente en `http://localhost:8501`.

## Subir a GitHub

```bash
git init
git add .
git commit -m "App interactiva Dcp: Al, Hg, Nb, Ta, Pb"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

La carpeta `Datos/` (~600 KB) se sube tal cual dentro del repositorio: es
pequeña y así la app funciona igual en local y en la nube sin pasos
adicionales.

## Desplegar en Streamlit Community Cloud (para compartir el enlace)

1. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión
   con tu cuenta de GitHub.
2. Haz clic en **"New app"**.
3. Elige tu repositorio, la rama `main` y el archivo principal `app.py`.
4. Haz clic en **"Deploy"**. En un par de minutos obtienes un enlace público
   (`https://<algo>.streamlit.app`) que puedes compartir con cualquier
   persona; no necesitan instalar nada.
5. Cada vez que hagas `git push` a `main`, la app se actualiza sola.

## Qué incluye la app

- **Análisis por elemento**: selector de elemento (Al, Hg, Nb, Ta, Pb) con
  las cinco gráficas interactivas ($\alpha^2F$, $N_{ph}$, $N_e$, $D_{cp}$ y
  las distribuciones electrónicas $\tilde D_{cp}$), métricas clave y
  descarga de los datos calculados en CSV.
- **Comparación global**: tabla de resultados, ajuste lineal
  $\tfrac12\Delta_{D_{cp}}$ vs. brecha experimental (con opción de excluir
  Nb, como en el informe), gráfico de dispersión teoría/experimento/cálculo
  y tabla de $\lambda$ (constante de acoplamiento electrón-fonón).
- **Literatura**: tabla filtrable de todas las referencias de $\Delta_0$
  usadas para cada elemento.
- **Metodología**: derivación resumida de la factorización de
  $D_{cp}=A(\omega)B(\omega)W(\omega)$ y detalles de la implementación
  numérica.
- Parámetros numéricos ajustables desde la barra lateral (resolución de la
  malla en $\omega$ y en energía electrónica, uso de $\omega_c$ de tabla o
  calculado por extrapolación).

## Créditos de los datos

Salidas DFT/DFPT en formato Quantum ESPRESSO brindadas por el profesor
Guillermo González. Formalismo de $D_{cp}(\omega,T_c)$ según sus notas de
cátedra (Universidad del Tolima, 2026).
