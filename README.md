# Minería de Datos — DevBugs

Sistema de recomendación de películas con **MovieLens ml-latest-small (100K)** y metodología CRISP-DM.

Ingeniería de Sistemas — Universidad Franz Tamayo · Docente: Enrique Alejandro Laurel Cossio

## Datos y entregas

La fuente del proyecto es [MovieLens ml-latest-small](anexo_ml-latest-small/ml-latest-small/), con **100.836 calificaciones** de 610 usuarios. La carpeta contiene las calificaciones, el catálogo de películas, las etiquetas y los enlaces.

| Actividad | Archivo |
|---|---|
| Análisis exploratorio del Hito 2 | [Cuaderno de EDA](anexo_ml-latest-small/evaluacion_procesual_hito_2.ipynb) |
| Evaluación grupal CRISP-DM (H3) | [Cuaderno grupal](h3_03_evaluacion1_grupal_proyecto_crisp.ipynb) |
| Respuestas a las 10 preguntas | [Cuaderno de respuestas](eval_respuestas_10_preguntas.ipynb) |
| Clasificación con regresión logística | [Cuaderno de regresión](WRANGLER/07_regresion_logistica/regresion_logistica.ipynb) |
| Clasificación con árbol de decisión | [Cuaderno del árbol](WRANGLER/08_arbol_decision/arbol_decision.ipynb) · [Versión del anexo](anexo_ml-latest-small/arbol_decision_movielens.ipynb) |
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

El entrenamiento genera el modelo y las métricas en `WRANGLER/04_modelado/resultados/`. Para ver el panel, inicia `python -m http.server` en la raíz y abre `http://localhost:8000/WRANGLER/04_modelado/resultados.html`. Para los cuadernos, abre la carpeta del repositorio en VS Code y elige el kernel de `.venv`; cada cuaderno usa rutas relativas a su propia carpeta.

## Historial

Hasta el commit `856ab7ad` el proyecto usaba MovieLens 1M. Desde la rama `devlop` todo el repositorio usa ml-latest-small: los cuadernos de clasificación se volvieron a ejecutar sin las variables de sexo y edad, que este conjunto de datos no incluye.
