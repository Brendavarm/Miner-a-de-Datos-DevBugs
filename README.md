# Minería de Datos — DevBugs

**Sistema de recomendación de películas con MovieLens 1M**, siguiendo la metodología CRISP-DM.

Carrera de Ingeniería de Sistemas — Universidad Franz Tamayo · Docente: Enrique Alejandro Laurel Cossio

## Entregas

| Tarea | Archivo |
|---|---|
| Proyecto DM, Fase 3-4 (evaluación 1 grupal) | [`h3_03_evaluacion1_grupal_proyecto_crisp.ipynb`](h3_03_evaluacion1_grupal_proyecto_crisp.ipynb) |
| Regresión logística para clasificación | [`WRANGLER/07_regresion_logistica/regresion_logistica.ipynb`](WRANGLER/07_regresion_logistica/regresion_logistica.ipynb) |
| Árbol de decisión para clasificación | [`WRANGLER/08_arbol_decision/arbol_decision.ipynb`](WRANGLER/08_arbol_decision/arbol_decision.ipynb) |
| Presentación (Canva) | [`enlace_canva.txt`](enlace_canva.txt) |

Los dos modelos de clasificación responden la misma pregunta (*¿le gustará la película al usuario?*, rating ≥ 4) con las mismas 23 variables y la misma partición 80/20, así que se pueden comparar directamente:

| Modelo | Exactitud (test) | AUC (test) |
|---|---:|---:|
| Línea base (decir "Me gusta" a todo) | 57,52% | 0,500 |
| Regresión logística | 72,00% | 0,782 |
| Árbol de decisión (profundidad 8) | 72,00% | 0,782 |

## Datos

Todo el proyecto usa **MovieLens 1M** (GroupLens): 1.000.209 calificaciones de 6.040 usuarios sobre 3.706 películas, entre 2000 y 2003. Los datos ya limpios están en [`WRANGLER/03_preparacion_datos/datos/`](WRANGLER/03_preparacion_datos/datos/).

El trabajo anterior con el dataset pequeño (ml-latest-small, 100.836 calificaciones) quedó como referencia en [`anexo_ml-latest-small/`](anexo_ml-latest-small/).

## Estructura

```text
h3_03_evaluacion1_grupal_proyecto_crisp.ipynb   entregable de la evaluación grupal (10 preguntas)
WRANGLER/
  00_ESTRUCTURA.txt          qué hay en cada carpeta
  01_comprension_negocio/    objetivo, tipo de tarea, variable y métrica
  02_comprension_datos/      EDA
  03_preparacion_datos/      datos limpios de MovieLens 1M
  04_modelado/               recomendador: baselines y factorización matricial (RMSE)
  05_evaluacion/             gráficos de evaluación
  06_presentacion/           guía de estudio
  07_regresion_logistica/    clasificación con regresión logística
  08_arbol_decision/         clasificación con árbol de decisión
anexo_ml-latest-small/       trabajo anterior con el dataset de 100k
```

## Cómo ejecutar

Requiere Python 3.12 o superior.

```bash
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Después, abrir el cuaderno, elegir el kernel `.venv` y usar *Ejecutar todo*. Cada cuaderno se ejecuta en menos de un minuto.
