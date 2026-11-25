# Minería de Datos — DevBugs

Sistema de recomendación de películas con **MovieLens ml-latest-small (100K)** y metodología CRISP-DM.

Ingeniería de Sistemas — Universidad Franz Tamayo · Docente: Enrique Alejandro Laurel Cossio

## Datos y entregas

La fuente del proyecto es [MovieLens ml-latest-small](anexo_ml-latest-small/ml-latest-small/), con **100.836 calificaciones** de 610 usuarios. La carpeta contiene las calificaciones, el catálogo de películas, las etiquetas y los enlaces.

| Actividad | Archivo |
|---|---|
| Análisis exploratorio del Hito 2 | [Cuaderno de EDA](anexo_ml-latest-small/evaluacion_procesual_hito_2.ipynb) |
| Clasificación con árbol de decisión | [Cuaderno de clasificación](anexo_ml-latest-small/arbol_decision_movielens.ipynb) |
| Segmentación con K-means | [Cuaderno del Hito 4](anexo_ml-latest-small/h4_1_kmeans_movielens.ipynb) |
| Recomendador y evaluación | [Entrenamiento](WRANGLER/04_modelado/scripts/00_entrenar_todo.py) · [Panel](WRANGLER/04_modelado/resultados.html) |
| Presentación | [Canva](enlace_canva.txt) |

## Cómo ejecutar

En PowerShell, desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe WRANGLER/04_modelado/scripts/00_entrenar_todo.py
```

El entrenamiento genera el modelo y las métricas en `WRANGLER/04_modelado/resultados/`. Para ver el panel, inicia `python -m http.server` en la raíz y abre `http://localhost:8000/WRANGLER/04_modelado/resultados.html`. Para ejecutar los cuadernos de 100K, abre `anexo_ml-latest-small/` en VS Code y elige el kernel de `.venv`.

## Entregas previas con 1M

El cuaderno grupal H3, la regresión logística y el árbol de decisión en `WRANGLER/07_regresion_logistica/` y `WRANGLER/08_arbol_decision/`, y los informes anteriores con cifras de 1M documentan esa etapa del proyecto. Para reproducirlos con sus datos originales, usa el commit `856ab7ad`. Para la versión actual de 100K, usa los cuadernos y el entrenamiento de la tabla.
