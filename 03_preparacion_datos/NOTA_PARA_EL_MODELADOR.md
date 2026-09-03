**Nota para el modelador **  
Ya terminé la parte de preparación (rol 3). Te dejo todo listo en 03_preparacion_datos/datos/  
   
 para que arranques con 04_modelado/scripts/03_modelo.py y 05_evaluacion/04_evaluacion.py.  
Léete al menos la sección "⚠️ Lo único que te puede arruinar el trabajo", que es  
   
 un error que se come a mucha gente y no da ningún síntoma hasta que las métricas  
   
 salen raras.  
**Lo que te dejo**  
| | |  
|-|-|  
| **Archivo** | **Para qué lo vas a usar** |   
| datos_limpios.parquet | **Para ** **surprise** **.** Es el que necesitas para entrenar. 1.000.209 ratings |   
| datos_limpios.csv | Lo mismo pero en CSV, por si prefieres verlo a ojo. Pesa 94 MB en vez de 13 |   
| matriz_usuario_pelicula.npz | **Para el mapa de calor de similitud.** Matriz dispersa 6.040 × 3.706 |   
| mapa_peliculas.csv | **Para el Top-N.** Traduce columna → movieId + título + géneros |   
| mapa_usuarios.csv | Traduce fila → userId |   
| matriz_usuario_pelicula.parquet | La matriz en versión densa, con NaN en los huecos. Cómoda para trastear |   
| peliculas_limpias.csv / usuarios.csv | Catálogo y datos demográficos (sexo, edad, ocupación) |   
| reporte_preparacion.json | Todas las métricas del proceso, por si necesitas citar algo en el informe |   
   
Dataset: **MovieLens 1M** — 6.040 usuarios, 3.706 películas, 1.000.209 calificaciones  
   
 entre abril de 2000 y febrero de 2003. Escala de 1 a 5 estrellas enteras (sin medias).  
**Lo único que te puede arruinar el trabajo**  
**En la matriz dispersa, un ** **0** ** significa "no ha visto esa película", NO "le puso cero estrellas".**  
Es así por cómo funciona el formato CSR: solo guarda las celdas con valor, y todo lo  
   
 demás lo devuelve como cero. Pero el 95,53% de la matriz son huecos, así que si haces  
   
 M.mean() estarás promediando 21 millones de ceros falsos y te va a salir una media  
   
 de 0,16 en vez de 3,58.  
Para calcular bien, enmascara los huecos:  
vistas = M.getnnz(axis=1)              # cuántas películas vio cada usuario  
 medias = M.sum(axis=1).A1 / vistas     # su media REAL, solo sobre lo que vio  
   
En matriz_usuario_pelicula.parquet esto no pasa: ahí los huecos son NaN de verdad  
   
 y pandas los ignora solo.  
**Cómo cargar todo (copia y pega)**  
import pandas as pd, scipy.sparse as sp  
   
 # Para entrenar con surprise:  
 df = pd.read_parquet("03_preparacion_datos/datos/datos_limpios.parquet")  
   
 # Para el mapa de calor de similitud:  
 M     = sp.load_npz("03_preparacion_datos/datos/matriz_usuario_pelicula.npz")  
 items = pd.read_csv("03_preparacion_datos/datos/mapa_peliculas.csv")   # columna -> título  
 users = pd.read_csv("03_preparacion_datos/datos/mapa_usuarios.csv")    # fila    -> userId  
   
**Ojo con ** **surprise** **: no acepta el ** **.npz** **.** Usa su propio formato y necesita el  
   
 parquet (o el CSV). Se carga así:  
from surprise import Dataset, Reader  
 reader = Reader(rating_scale=(1, 5))  
 data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)  
   
Ya lo probé y funciona: un SVD() con los parámetros por defecto entrena en **6,5**  
 **  
 segundos** y da  **RMSE 0.8729 / MAE 0.6845**. Te sirve como referencia — si tus  
   
 métodos dan mucho peor que eso, algo va mal en el montaje.  
También te confirmo que **scikit-surprise** ** sí instala en Python 3.12** (versión  
   
 1.1.5, con pip install scikit-surprise). Lo digo porque esa librería daba  
   
 problemas de instalación en versiones anteriores y podrías perder un rato ahí.  
**De dónde sale cada gráfico tuyo**  
| | |  
|-|-|  
| **Tu gráfico** | **De dónde** |   
| Mapa de calor de similitud | **Directo de mi ** **.npz**, con cosine_similarity. Probado, funciona |   
| Top-N recomendado a un usuario | De tu modelo entrenado + mapa_peliculas.csv para poner los títulos |   
| Comparación de RMSE entre métodos | De entrenar varios modelos y compararlos — eso ya es cosa tuya |   
| Curva de precisión@K | De tu evaluación |   
   
Para el mapa de calor, un aviso práctico: **no intentes dibujar los 6.040 usuarios**.  
   
 La matriz de similitud completa sería de 36 millones de celdas y no se ve nada. Coge  
   
 una muestra de 100–300 y ya se entiende la idea. Yo hice algo parecido en  
   
 03_preparacion_datos/graficos/dispersion_matriz.png, échale un ojo si quieres.  
**Tres cosas que encontré y que te van a afectar**  
**1. La matriz está vacía al 95,53%.** De 22,4 millones de celdas posibles, solo 1  
   
 millón tiene valor. Es el problema central de tu modelo, y el número que más vas a  
   
 repetir en la presentación.  
**2. La gente califica alto: la media es 3,58.** El 57% de los votos son 4 o 5  
   
 estrellas y solo el 5,6% son de 1. Esto significa que un modelo tonto que prediga  
   
 "3,58 siempre" ya saca un RMSE decente. Si no normalizas restando la media de cada  
   
 usuario, tu modelo bueno no se va a distinguir del tonto. Merece la pena que metas  
   
 ese predictor tonto como línea base en tu comparación de RMSE — queda muy bien en el  
   
 informe y demuestra que tu modelo aporta algo.  
**3. Cola larga bestial.** El 20% de usuarios más activos concentra el 57% de los  
   
 votos, y el 20% de películas más populares el 65%. Hay usuarios con 2.314  
   
 calificaciones y películas con **una sola**. Con 1 voto no se puede calcular una  
   
 similitud fiable, así que probablemente tengas que filtrar por un mínimo (unos 5 o 10  
   
 votos) antes de recomendar. Los usuarios no dan problema: MovieLens ya garantiza que  
   
 todos tienen 20 o más.  
**Un consejo sobre el split**  
Cuando partas en entrenamiento y prueba: **si normalizas o filtras, hazlo usando solo**  
 **  
 el conjunto de entrenamiento.** Si calculas la media de un usuario con calificaciones  
   
 que están en el test, eso es fuga de datos y te va a inflar los resultados. Es de las  
   
 cosas que el docente puede detectar y resta bastante.  
Y para el split en sí, mejor que sea por usuario (dejar fuera unas cuantas  
   
 calificaciones de cada uno) y no puramente aleatorio, para que todos los usuarios  
   
 tengan algo que evaluar. Te dejé la columna fecha en los datos por si quieres  
   
 hacerlo temporal: entrenar con lo viejo y evaluar con lo reciente, que es lo más  
   
 realista.  
**Para ejecutar**  
Las librerías están en un entorno virtual dentro del proyecto. Úsalo así:  
./.venv/bin/python 04_modelado/scripts/03_modelo.py  
   
Si python3 04_modelado/scripts/03_modelo.py a secas te falla diciendo que no encuentra pandas, es por  
   
 esto. Si trabajas en tu propia máquina, necesitas:  
python3 -m venv .venv  
 ./.venv/bin/pip install pandas numpy scipy matplotlib pyarrow scikit-learn scikit-surprise  
   
Si necesitas que te exporte algo en otro formato, o que te deje el split de  
   
 entrenamiento/prueba ya hecho, dime y te lo preparo — es rápido.  
Todo esto lo genera 02_preparacion.py. Si quieres regenerarlo desde cero:  
   
 ./.venv/bin/python 02_preparacion.py --datos ./ml-1m  
