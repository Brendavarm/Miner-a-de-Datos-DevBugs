# Datos del recomendador

La fuente es [MovieLens ml-latest-small](../../anexo_ml-latest-small/ml-latest-small/). Los scripts de modelado leen `ratings.csv` para entrenar y `movies.csv` para mostrar los títulos. El archivo de calificaciones contiene 100.836 filas con `userId`, `movieId`, `rating` y `timestamp`.

La evaluación compara los modelos de referencia y la factorización matricial con la misma división de entrenamiento y prueba. Para generar métricas y el modelo ejecuta `python WRANGLER/04_modelado/scripts/00_entrenar_todo.py` desde la raíz.
