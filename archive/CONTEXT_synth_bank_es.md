> **Superseded (2026-11-22):** this is the hand-over note for the original *synth-bank-es* credit-risk project. For RiskSynth AI see `README.md` and `docs/MIGRATION_RISKSYNTH.md`.

# CONTEXT.md — synth-bank-es

Documento de retoma. Destilado del repo a 2026-08-28. Lo no verificable está
marcado como **PENDIENTE DE CONFIRMAR**.

## 1. Qué es

Pipeline para generar **datasets bancarios sintéticos realistas del mercado
español**, calibrados con microdatos oficiales del INE (ECV, EPF, EPA). El
objetivo es disponer de datos para desarrollo de modelos de riesgo de crédito
(scoring, predicción de default), pruebas de cumplimiento regulatorio e
investigación ML sin usar datos de clientes reales (restricción GDPR). El
resultado busca ser estadísticamente indistinguible de datos bancarios reales
pero sin individuos reales.

## 2. Stack

- **Lenguaje:** Python 3.11 (pyproject declara `>=3.10`; README y quick start usan
  conda con 3.11).
- **Datos:** pandas ≥2.0, numpy, scipy, pyarrow/parquet, openpyxl/xlrd.
- **Generación sintética:** SDV ≥1.9 (CTGAN, GaussianCopula, TVAE/VAE), ctgan.
- **Deep learning:** tensorflow, torch, keras listados en `requirements.txt`
  (uso real limitado; **PENDIENTE DE CONFIRMAR** qué componente los necesita —
  el generador basado en reglas y el logit no).
- **Evaluación:** scipy.stats (KS, chi², Wasserstein), Jensen-Shannon.
- **Web:** Streamlit ≥1.28 + Plotly.
- **Utilidades:** loguru, pyyaml, python-dotenv, tqdm, joblib, multiprocessing.
- **Calidad/dev:** pytest + pytest-cov, black (line-length 100), flake8, mypy.
- **Despliegue:** ninguno configurado. Ejecución local (CLI + `streamlit run
  streamlit_app.py` en localhost:8501). Sin Docker, sin CI. **PENDIENTE DE
  CONFIRMAR** si hay intención de desplegar (p. ej. Streamlit Cloud).

## 3. Arquitectura

Pipeline de 5 etapas, orquestadas manualmente (no hay runner único):

1. **Ingesta.** Descarga manual de microdatos INE a `data/raw/{ecv,epf,epa}/`
   (ficheros `.tab`, no versionados en git). ECV 2024, EPF 2024, EPA 2025 T3.
   Directorios `src/data/ingestion/` y `src/data/cleaning/` existen vacíos
   (placeholders para extender).

2. **Procesado.** `src/data/processing/process_data.py` (~311 líneas): detecta
   separador, mapea códigos de columna INE a nombres legibles (p. ej. ECV
   `PB140→edad`, `HY020→renta`), y fusiona las tres encuestas. Salida:
   `data/processed/*.parquet` (ecv, epf, epa y `combined_base.parquet` como base
   de entrenamiento). **Los parquet SÍ están versionados** en el repo (~11 MB) para
   poder generar sin reprocesar.

3. **Generación.** Dos vías independientes:
   - **A — Basado en reglas + multiproceso** (`generate_synthetic.py`, ~1134
     líneas). CLI interactiva (nº registros, tasa de default objetivo, cores,
     batch size). Parte de `combined_base.parquet` y sintetiza la jerarquía
     bancaria completa por cliente: identidad España-specific (nombre, DNI, IBAN,
     email, teléfono, dirección, CP por CCAA), ingresos/gastos/ahorro por CCAA,
     cuentas, tarjetas, préstamos (hipotecario/consumo con amortización),
     inversiones, seguros, 12 meses de transacciones categorizadas, contratos e
     historial. Calcula P(default) con un **modelo logit calibrado**
     (`calcular_prob_default_logit` + `COEF_LOGIT`): ratio de endeudamiento,
     log-ingresos, log-balance, tasa de ahorro negativa, tramos de edad,
     hipoteca/préstamo e interacción "doble deuda", riesgo regional por CCAA de
     alto AROPE, ratio gastos/ingresos, falta de colchón, "joven + pobre", ocio
     excesivo. Permite fijar tasa de default exacta. Salida: CSV plano en
     `data/synthetic/`.
   - **B — Modelo generativo SDV** (`client_generator.py`, ~332 líneas). Clase
     `ClientGenerator(n_samples, method, seed)`; `method ∈ {ctgan, copula, vae}`.
     Carga datos procesados, construye metadata SDV, entrena, aplica restricciones
     de negocio post-generación (edad 18–~100, saldos no negativos, DTI ≤40%,
     coherencia edad-empleo, importes mínimos de préstamo), y guarda salida
     versionada: CSV + `.pkl` del modelo + JSON de metadatos (timestamp, método,
     seed, shape).
   - `src/synthetic/{models,constraints,utils}/` vacíos (placeholders).

4. **Evaluación.** `src/evaluation/statistical_tests.py` (~392 líneas). Clase
   `StatisticalTests(real, synthetic, alpha)`: KS (numéricas), chi² (categóricas),
   distancia de Wasserstein, divergencia Jensen-Shannon, comparación de matriz de
   correlación y estadística descriptiva lado a lado. Genera informe HTML en
   `reports/evaluation_reports/`. Umbrales de calidad por Wasserstein
   (<0.05 excelente … >0.20 pobre).

5. **Interfaz web.** `streamlit_app.py` (~480 líneas). Pestañas: Dashboard
   (métricas), Data Generator (sube datos, elige método, genera y descarga),
   Evaluation (sube real+sintético, corre tests, gráficas comparativas), Settings
   (hiperparámetros: epochs, batch size, latent dim).

**Configuración:** `configs/data_config.yaml` (fuentes INE/Banco de España,
rutas y retención, reglas de calidad, encoding/escalado, GDPR) y
`configs/model_config.yaml` (hiperparámetros CTGAN/VAE/copula/TVAE, un
`timeseries_gan` para transacciones marcado enabled pero **PENDIENTE DE
CONFIRMAR** si tiene implementación, reglas de negocio, umbrales de evaluación).
Variables de entorno vía `.env` (copiar de `.env.example`): `INE_API_KEY`,
`BDE_API_KEY`, rutas, seed 42, `CTGAN_EPOCHS=300`, `VAE_LATENT_DIM=128`.

**Scoring:** `src/scoring/` existe vacío. Hay un notebook `credit_scoring.ipynb`
y `01_eda_N_50K.ipynb` (EDA sobre muestra sintética de 50K) en
`notebooks/02_data_processing/`. **PENDIENTE DE CONFIRMAR** el estado de ese
trabajo (no hay módulo de scoring en `src/`).

## 4. Decisiones clave

- **Calibrar con microdatos reales del INE en lugar de distribuciones
  inventadas.** ECV (condiciones de vida: edad, estudios, renta, situación
  laboral, hogar), EPF (presupuestos: gasto por categoría, régimen de vivienda,
  tamaño del hogar), EPA (población activa: ocupación, sector, tramo salarial).
  Motivo: realismo estadístico y defensa frente a auditoría.
- **Versionar los parquet procesados en git** (~11 MB) pese a ser datos
  derivados, para que cualquiera genere sintéticos sin descargar ni reprocesar
  los microdatos INE (que son grandes y en formatos SPSS/SAS/STATA). Los `raw/`
  quedan fuera de git.
- **Dos generadores en paralelo, no uno.** Reglas+multiproceso para escala grande
  y control fino (tasa de default exacta, jerarquía bancaria completa);
  SDV/CTGAN para fidelidad estadística multivariante aprendida. Se mantienen
  ambos en vez de elegir.
- **Modelo de default como logit calibrado a mano** (14 factores con
  interacciones) en la vía basada en reglas, en lugar de entrenar un clasificador.
  Según el histórico de conversaciones (`notas de recuperacion del proyecto
  .mkd`), el generador actual desciende de
  `generator_wealthreader_v6_2.py` (v6.2), que integra un "Nexus Score" y factores
  de default calibrados con literatura científica y fuentes INE (ECV, EPF, EES,
  EPA, Censo, ECH) + Banco de España (EFF 2022). **PENDIENTE DE CONFIRMAR** cuánto
  de esa v6.2 llegó íntegro al `generate_synthetic.py` de este repo.
- **SDV como framework generativo** (CTGAN para mixto num/cat, GaussianCopula
  como baseline rápido con correlaciones, VAE/TVAE deep). Alternativa descartada
  implícita: implementar GANs a mano.
- **Restricciones de negocio aplicadas post-generación**, no como
  condicionamiento del modelo (más simple, independiente del método).
- **Streamlit** para interfaz de usuarios no técnicos en vez de API/servicio.
- Config declarativa en YAML separada de código; `data_config.yaml` referencia
  fuentes previstas no usadas todavía (Padrón, EFF de Banco de España, CIRBE) —
  **PENDIENTE DE CONFIRMAR** si son roadmap o restos.

## 5. Estado actual

- **Rama:** `main`, al día con `origin/main`. **Un único commit**: `b709701
  "Initial commit: synth-bank-es pipeline"`. No hay otras ramas.
- **Sin commitear (untracked):**
  - `assets/` → `assets/images/logo.png` (logo, probablemente para el README /
    Streamlit).
  - `notas de recuperacion del proyecto .mkd` (~117 KB): registro de
    chat sobre la recuperación/linaje del generador (`generator_wealthreader_v6_2`).
    Es material de referencia; decidir si se versiona, se mueve a `docs/` o se
    ignora.
- **Funciona (según código y README, no verificado en ejecución aquí):** procesado
  INE→parquet; generador por reglas con CLI y multiproceso; generador SDV;
  suite de tests estadísticos + informe HTML; app Streamlit. Los 4 parquet
  procesados están presentes en `data/processed/`.
- **Vacío / placeholder:** `src/data/{ingestion,cleaning}`,
  `src/synthetic/{models,constraints}`, `src/scoring`, `src/visualization`,
  `src/utils` (solo `__init__.py`).
- **Tests:** estructura de carpetas en `tests/` (`test_scoring`, `test_data`,
  `test_generators`, `test_evaluation`) pero **sin archivos `test_*.py`** — no hay
  tests reales pese a la config de pytest/cov.
- **No hay** `DECISIONS.md`, `TODO.md`, ni `JOURNAL/`/`docs/`. README es la única
  documentación.
- **Inconsistencias detectadas:**
  - `pyproject.toml` tiene autor placeholder (`tu_email@example.com`) y no lista
    scipy/streamlit/plotly (sí están en `requirements.txt`).
  - README documenta método `'vae'` pero `model_config.yaml` define además `tvae`
    (deshabilitado) — aclarar qué mapea SDV realmente.
  - `requirements.txt` arrastra tensorflow+torch+keras; uso real sin confirmar.

## 6. Próximos pasos (priorizados)

1. **Decidir destino de los 2 untracked** y commitear: mover el `.mkd` a `docs/`
   (o `.gitignore`), añadir `assets/images/logo.png`.
2. **Escribir tests mínimos** en `tests/` (al menos: `process_data`,
   `calcular_prob_default_logit`, restricciones de `client_generator`,
   `StatisticalTests`), ya que la infra de pytest/cobertura está lista y vacía.
3. **Arreglar metadatos de packaging:** autor/email reales en `pyproject.toml`,
   alinear dependencias entre `pyproject.toml` y `requirements.txt`, y depurar
   tensorflow/torch/keras si no se usan.
4. **Verificar end-to-end** cada etapa en un entorno limpio (conda 3.11) y anotar
   comandos reales; crear un runner o Makefile que encadene process→generate→
   evaluate.
5. **Consolidar el linaje del generador:** documentar en `DECISIONS.md` qué es el
   "Nexus Score", los 14 factores y su calibración (fuente
   `generator_wealthreader_v6_2.py` / conversaciones), y confirmar paridad con
   `generate_synthetic.py`.
6. **Cerrar el módulo de scoring:** decidir si el trabajo de
   `credit_scoring.ipynb` se productiviza en `src/scoring/` o se deja como
   notebook, y documentarlo.
7. **Limpiar `data_config.yaml`:** marcar fuentes no implementadas (Padrón, EFF,
   CIRBE) como roadmap explícito o eliminarlas.
8. **Definir despliegue** de la app Streamlit si procede (Streamlit Cloud/Docker).
