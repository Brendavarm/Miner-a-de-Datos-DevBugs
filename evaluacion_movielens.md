# Evaluación grupal del proyecto de minería de datos

## Proyecto: MovieLens ml-latest-small

Este proyecto utiliza el conjunto de datos MovieLens para predecir si la calificación de una película será alta o baja. El dataset contiene 100.836 calificaciones, 9.742 películas, 3.683 etiquetas y 610 usuarios. [file:6]

## 1. Entendimiento del negocio

El proyecto aborda la necesidad de identificar si una película recibirá una calificación alta.

La decisión que busca apoyar es priorizar películas potencialmente bien valoradas para mejorar una futura recomendación o selección de contenido.

El análisis afecta principalmente a los usuarios, porque pueden recibir recomendaciones más relevantes, y a la plataforma, porque puede ordenar o filtrar películas según su valoración esperada.

Se definió como calificación alta aquella mayor o igual a 4,0 estrellas:

- **Clase 1:** rating alto, `rating >= 4.0`.
- **Clase 0:** rating bajo, `rating < 4.0`.

El criterio de éxito consiste en obtener un modelo capaz de clasificar correctamente las calificaciones y superar una predicción básica de la clase mayoritaria.

## 2. Métrica principal

La métrica principal seleccionada es el **recall de la clase positiva**, porque permite medir qué proporción de las calificaciones realmente altas fue detectada por el modelo.

$$
Recall = \\frac{VP}{VP + FN}
$$

Donde:

- `VP`: verdaderos positivos.
- `FN`: falsos negativos.
- Unidad: proporción o porcentaje de calificaciones altas.

En el conjunto de prueba:

$$
Recall = \\frac{6788}{6788 + 2928} = 0.6986
$$

El modelo detectó correctamente el **69,86 % de las calificaciones altas**.

| Métrica | Resultado |
|---|---:|
| Accuracy | 65,81 % |
| Balanced accuracy | 65,95 % |
| Precision | 63,11 % |
| Recall | 69,86 % |
| F1-score | 66,31 % |
| ROC-AUC | 72,64 % |
| Average precision | 69,54 % |

El recall es una métrica de desempeño del modelo, no una métrica de negocio observada directamente. Una acción concreta para mejorarla sería ajustar el umbral de decisión o comparar el árbol de decisión con regresión logística utilizando el mismo procedimiento de evaluación.

## 3. Entendimiento de los datos

La fuente utilizada es el conjunto de datos **MovieLens ml-latest-small**.

Cada fila de `ratings.csv` representa la calificación de una película realizada por un usuario. Sus variables son:

- `userId`: identificador anónimo del usuario.
- `movieId`: identificador de la película.
- `rating`: calificación entre 0,5 y 5,0 estrellas.
- `timestamp`: fecha y hora de la calificación.

| Archivo | Registros | Variables |
|---|---:|---:|
| `ratings.csv` | 100.836 | 4 |
| `movies.csv` | 9.742 | 3 |
| `tags.csv` | 3.683 | 4 |
| `links.csv` | 9.742 | 3 |

Para el modelado se unieron `ratings.csv` y `movies.csv` mediante la columna `movieId`.

Después de la preparación se utilizaron **100.836 observaciones válidas** y cuatro variables predictoras:

- `genre_count`.
- `user_mean_rating`.
- `user_rating_count`.
- `year`.

## 4. Calidad y preparación de los datos

La revisión exploratoria encontró ocho valores faltantes en `links.csv`, columna `tmdbId`, equivalentes aproximadamente al 0,08 % de esa columna. No se encontraron cadenas vacías relevantes. [file:2]

No se eliminaron automáticamente los valores atípicos estadísticos, porque las calificaciones entre 0,5 y 5,0 son valores válidos del dataset. [file:2]

### Tratamientos aplicados

1. Se cargaron `ratings.csv` y `movies.csv`.
2. Se creó la variable objetivo `rating_alto`.
3. Se unieron las calificaciones con los géneros de cada película usando `movieId`.
4. Se reemplazaron valores faltantes de `genres` por una cadena vacía.
5. Se creó `genre_count`, que indica la cantidad de géneros de una película.
6. Se calculó `user_mean_rating`, que representa el promedio histórico de calificación de cada usuario.
7. Se calculó `user_rating_count`, que representa la cantidad de calificaciones realizadas por cada usuario.
8. Se convirtió `timestamp` a fecha.
9. Se extrajo el año en la variable `year`.
10. Se eliminaron filas con valores faltantes en las variables utilizadas por el modelo.

### Limitación metodológica

Las estadísticas del usuario se calcularon antes de dividir los datos en entrenamiento y prueba. Esto puede producir fuga de información, porque el promedio y la cantidad de ratings pueden incluir datos que pertenecen al conjunto de prueba.

La mejora recomendada consiste en calcular las estadísticas utilizando únicamente el conjunto de entrenamiento y aplicar después esa transformación al conjunto de prueba.

## 5. Resultado interesante de la exploración

Se encontró una relación positiva entre la cantidad de ratings realizados por un usuario y la cantidad de etiquetas que crea.

La correlación obtenida fue:

$$
r = 0.3579
$$

Esto indica una relación positiva moderada: los usuarios que califican más películas tienden también a crear más etiquetas. [file:2]

### Gráfico recomendado

**Título:** Relación entre ratings y etiquetas por usuario

- Eje X: Ratings realizados por usuario.
- Eje Y: Tags creados por usuario.
- Fuente: MovieLens ml-latest-small.

Este resultado es importante porque muestra que la actividad del usuario podría ser útil para construir variables predictoras. Sin embargo, la correlación no demuestra causalidad.

# Modelado

## 6. Variable objetivo y modelo utilizado

La variable objetivo es `rating_alto`:

```python
rating_alto = (rating >= 4.0).astype(int)
```

| Clase | Interpretación |
|---:|---|
| 0 | Rating bajo: menor que 4,0 |
| 1 | Rating alto: mayor o igual a 4,0 |

Se utilizó un **árbol de decisión**, uno de los modelos revisados en clase.

El modelo es pertinente porque permite resolver un problema de clasificación binaria, no requiere normalizar las variables numéricas, produce reglas interpretables y permite visualizar cómo se toman las decisiones.

La configuración utilizada fue:

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42
)
```

`max_depth=6` y `min_samples_leaf=50` ayudan a limitar la complejidad del modelo y reducir el sobreajuste.

## 7. Partición de los datos

| Conjunto | Proporción | Registros |
|---|---:|---:|
| Entrenamiento | 80 % | 80.668 |
| Prueba | 20 % | 20.168 |

Se utilizó una separación estratificada con `random_state=42`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

La opción `stratify=y` conserva aproximadamente la misma proporción de calificaciones altas y bajas en ambos conjuntos.

El conjunto de prueba se utilizó únicamente después del entrenamiento para medir el desempeño final.

Como limitación, las estadísticas de usuario fueron calculadas antes de la partición. Para evitar completamente la fuga de información, estas estadísticas deberían calcularse únicamente con los datos de entrenamiento.

## 8. Variables más importantes

La importancia de las variables se obtuvo mediante:

```python
tree_model.feature_importances_
```

| Variable | Importancia |
|---|---:|
| `user_mean_rating` | 0,9414 |
| `year` | 0,0413 |
| `user_rating_count` | 0,0140 |
| `genre_count` | 0,0034 |

La variable más importante fue `user_mean_rating`, con una importancia aproximada de **94,14 %**.

Esto significa que, dentro de las reglas aprendidas por el árbol, el promedio histórico del usuario fue la variable que más contribuyó a separar las clases.

Este resultado no demuestra causalidad. Solamente indica que la variable fue importante para este modelo y este conjunto de datos.

Una posible interpretación es que algunos usuarios tienden a asignar calificaciones más altas o más bajas que otros. Sin embargo, esta conclusión debe tomarse con precaución debido a la posible fuga de información.

## 9. Desempeño del modelo

La matriz de confusión obtenida en el conjunto de prueba fue:

| Valor real / Predicción | Bajo | Alto |
|---|---:|---:|
| Bajo | 6.484 | 3.968 |
| Alto | 2.928 | 6.788 |

Las métricas obtenidas fueron:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,6581 |
| Balanced accuracy | 0,6595 |
| Precision | 0,6311 |
| Recall | 0,6986 |
| F1-score | 0,6631 |
| ROC-AUC | 0,7264 |
| Average precision | 0,6954 |

El modelo clasificó correctamente el **65,81 %** de las observaciones del conjunto de prueba.

En este problema, el error más importante puede ser el falso negativo:

- Un falso negativo ocurre cuando una calificación realmente alta se predice como baja.
- Este error puede provocar que se descarte una película que podría ser útil para recomendar.

La matriz muestra **2.928 falsos negativos**. Por ello, si el objetivo es no perder películas potencialmente bien valoradas, se debería priorizar la reducción de estos errores mediante un mayor recall.

## 10. Visualización de las decisiones del modelo

La visualización principal es el árbol de decisión generado con:

```python
plot_tree(
    tree_model,
    feature_names=feature_cols,
    class_names=["Bajo", "Alto"],
    filled=True,
    rounded=True,
    max_depth=3
)
```

El árbol permite observar las condiciones utilizadas para separar las clases.

Una ruta de decisión puede interpretarse de la siguiente manera:

> Si el `user_mean_rating` supera un determinado umbral aprendido por el árbol, la observación se dirige hacia una rama con mayor proporción de ratings altos. Después, variables como `year` o `user_rating_count` pueden refinar la clasificación.

El umbral exacto debe leerse directamente en el gráfico generado por el notebook.

Esta visualización aporta interpretabilidad, porque muestra las reglas y condiciones que llevan a predecir una calificación como alta o baja.

## Corrección del notebook antes de entregar

El notebook contiene una salida de error relacionada con el paquete `ipykernel`. Debe instalarse en el entorno virtual:

```bash
/home/jasser/Documentos/MINERIA2/Miner-a-de-Datos-DevBugs/.venv/bin/python -m pip install ipykernel -U
```

Después:

1. Seleccionar el kernel `.venv`.
2. Ejecutar todas las celdas desde el inicio.
3. Confirmar que no existan errores.
4. Guardar el notebook ejecutado.
5. Verificar que las métricas y gráficos coincidan con la presentación.

## Resultados medidos

- **Accuracy:** 65,81 %.
- **Recall:** 69,86 %.
- **F1-score:** 66,31 %.
- **ROC-AUC:** 72,64 %.
- **Observaciones de entrenamiento:** 80.668.
- **Observaciones de prueba:** 20.168.
- **Variable más importante:** `user_mean_rating`, con 94,14 %.
