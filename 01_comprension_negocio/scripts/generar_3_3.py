# -*- coding: utf-8 -*-
"""
Genera el material de apoyo del punto 3.3:
la VARIABLE A OPTIMIZAR r(u,i) = columna 'rating'.

Salidas:
  01_comprension_negocio/graficos/3_3_variable_a_optimizar.png
  03_preparacion_datos/datos/muestra_rating.csv   (100 filas, abrible en Excel)
  03_preparacion_datos/datos/muestra_rating.txt   (lo mismo, en texto plano)

Uso:  ./.venv/bin/python 01_comprension_negocio/scripts/generar_3_3.py
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def miles(n):
    """1000209 -> '1.000.209'  (separador espanol)"""
    return f"{n:,}".replace(",", ".")

RAIZ = "."
PARQUET = f"{RAIZ}/03_preparacion_datos/datos/datos_limpios.parquet"

print("Cargando", PARQUET, "...")
df = pd.read_parquet(PARQUET)
print(f"  {len(df):,} filas, columnas: {list(df.columns)}")

# ---------------------------------------------------------------- MUESTRA
cols = ["userId", "movieId", "rating", "title", "fecha"]
muestra = df[cols].head(100).copy()
muestra.to_csv(f"{RAIZ}/03_preparacion_datos/datos/muestra_rating.csv",
               index=False, encoding="utf-8-sig")

with open(f"{RAIZ}/03_preparacion_datos/datos/muestra_rating.txt", "w",
          encoding="utf-8") as f:
    f.write("=" * 78 + "\n")
    f.write("MUESTRA DE datos_limpios.parquet  -  primeras 100 filas de 1.000.209\n")
    f.write("La columna 'rating' ES la variable a optimizar  r(u,i)\n")
    f.write("  userId  = u   (el usuario)\n")
    f.write("  movieId = i   (la pelicula)\n")
    f.write("  rating  = r(u,i)  <-- VARIABLE A OPTIMIZAR, escala 1 a 5\n")
    f.write("=" * 78 + "\n\n")
    f.write(muestra.to_string(index=False))
    f.write("\n\n" + "=" * 78 + "\n")
    f.write(f"Total real del fichero: {miles(len(df))} calificaciones\n")
    f.write(f"Media de rating: {df['rating'].mean():.4f}\n")
    f.write(f"Minimo: {df['rating'].min():.0f}   Maximo: {df['rating'].max():.0f}\n")
    f.write("=" * 78 + "\n")
print("  -> muestra_rating.csv / .txt")

# ---------------------------------------------------------------- CIFRAS
n_users = df.userId.nunique()
n_items = df.movieId.nunique()
total   = n_users * n_items
obs     = len(df)
vacias  = total - obs
dist    = df.rating.value_counts().sort_index()
media   = df.rating.mean()

# ---------------------------------------------------------------- FIGURA
fig = plt.figure(figsize=(16, 9.5))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05],
                      hspace=0.32, wspace=0.22,
                      left=0.05, right=0.96, top=0.88, bottom=0.06)

fig.suptitle("3.3  LA VARIABLE A OPTIMIZAR:  rating  r(u,i)",
             fontsize=20, fontweight="bold", y=0.965)
fig.text(0.5, 0.918,
         "MovieLens 1M  ·  1.000.209 calificaciones  ·  6.040 usuarios  ·  3.706 peliculas",
         ha="center", fontsize=11, color="#555555")

AZUL, ROJO, GRIS = "#1f77b4", "#d62728", "#888888"

# --- (A) forma larga: la tabla ---------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
ax.set_title("A)  Donde esta:  columna 'rating' de datos_limpios.parquet",
             fontsize=13, fontweight="bold", loc="left", pad=14)

filas = df[["userId", "movieId", "rating", "title"]].head(6).values
cab = ["userId\n(u)", "movieId\n(i)", "rating\nr(u,i)", "title"]
tabla = ax.table(
    cellText=[[f"{int(a)}", f"{int(b)}", f"{c:.0f}", (d[:26] + "…") if len(d) > 26 else d]
              for a, b, c, d in filas],
    colLabels=cab, cellLoc="center", loc="upper center",
    colWidths=[0.13, 0.14, 0.15, 0.58])
tabla.auto_set_font_size(False); tabla.set_fontsize(10.5); tabla.scale(1, 1.85)
for (r, c), cell in tabla.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#e8e8e8"); cell.set_text_props(fontweight="bold")
    if c == 2:                                   # resaltar la columna rating
        cell.set_facecolor("#ffd9d9" if r > 0 else "#f2a0a0")
        cell.set_text_props(fontweight="bold", color="#8b0000" if r > 0 else "black")
    if c == 3:
        cell.set_text_props(ha="left")
ax.text(0.5, 0.14,
        "Cada fila = una celda YA OBSERVADA.   Ejemplo:  r(1, 1193) = 5",
        ha="center", fontsize=11, style="italic", color="#333333",
        transform=ax.transAxes)
ax.text(0.5, 0.04, f"{miles(obs)} filas como estas  ({obs/total*100:.2f}% de la matriz)",
        ha="center", fontsize=10, color=GRIS, transform=ax.transAxes)

# --- (B) forma matriz: los huecos ------------------------------------
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.set_title("B)  Que hay que optimizar:  las celdas vacias",
             fontsize=13, fontweight="bold", loc="left", pad=14)

rng = np.random.default_rng(7)
nf, nc = 6, 8
ax.set_xlim(-1.55, nc + 0.4); ax.set_ylim(-1.5, nf + 0.5); ax.invert_yaxis()
for i in range(nf):
    for j in range(nc):
        hay = rng.random() < 0.28
        val = f"{rng.integers(1, 6)}" if hay else "?"
        ax.add_patch(Rectangle((j, i), .92, .88,
                               facecolor="#cfe3f3" if hay else "#fdecec",
                               edgecolor="#b0b0b0", lw=.8))
        ax.text(j + .46, i + .44, val, ha="center", va="center",
                fontsize=13 if hay else 15,
                fontweight="bold", color=AZUL if hay else ROJO)
    ax.text(-.25, i + .44, f"u{i+1}", ha="right", va="center", fontsize=10)
for j in range(nc):
    ax.text(j + .46, -.22, f"i{j+1}", ha="center", va="bottom", fontsize=10)
ax.text(-1.35, nf/2, "usuarios\n6.040", rotation=90, ha="center", va="center",
        fontsize=11, fontweight="bold")
ax.text(nc/2, -1.05, "peliculas   3.706", ha="center", fontsize=11, fontweight="bold")

ax.add_patch(Rectangle((0, nf + .35), .8, .55, facecolor="#cfe3f3", edgecolor="#b0b0b0"))
ax.text(1.0, nf + .63, f"observadas: {miles(obs)}  ({obs/total*100:.2f}%)  → entrenar",
        va="center", fontsize=10.5)
ax.add_patch(Rectangle((0, nf + 1.0), .8, .55, facecolor="#fdecec", edgecolor="#b0b0b0"))
ax.text(1.0, nf + 1.28, f"vacias: {miles(vacias)}  ({vacias/total*100:.2f}%)  → PREDECIR",
        va="center", fontsize=10.5, fontweight="bold", color=ROJO)

# --- (C) distribucion de la variable ---------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.set_title("C)  Como se distribuye la variable objetivo (lo observado)",
             fontsize=13, fontweight="bold", loc="left", pad=12)
barras = ax.bar([f"{int(k)}★" for k in dist.index], dist.values,
                color=["#c94b4b", "#e08a5a", "#d9c15a", "#7fb069", "#4a9d5f"],
                edgecolor="white", lw=1.5)
for b, v in zip(barras, dist.values):
    ax.text(b.get_x() + b.get_width()/2, v + 9000,
            f"{miles(v)}\n{v/obs*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="#333333", lw=1)
ax.set_ylabel("nº de calificaciones", fontsize=11)
ax.set_ylim(0, dist.max() * 1.22)
ax.spines[["top", "right"]].set_visible(False)
ax.text(.99, .93, f"media = {media:.2f} ★\nescala 1 a 5, enteros",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#bbbbbb"))
ax.text(.02, .93, "sesgada al alza:\n57,5% son 4★ o 5★\n(solo 5,6% son 1★)",
        transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
        style="italic", color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fdecec", ec="#e0a0a0"))

# --- (D) ficha resumen ------------------------------------------------
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
ax.set_title("D)  Ficha de la variable", fontsize=13, fontweight="bold",
             loc="left", pad=12)
ax.add_patch(Rectangle((0, 0), 1, .93, transform=ax.transAxes,
                       facecolor="#fafafa", edgecolor="#cccccc", lw=1.2))
ficha = [
    ("Nombre",              "rating   ·   r(u,i)"),
    ("Se lee",              "nota que el usuario u daria a la pelicula i"),
    ("Tipo",                "numerica  ·  continua en la prediccion"),
    ("Rango",               "[1, 5] estrellas"),
    ("Fichero",             "03_preparacion_datos/datos/datos_limpios.parquet"),
    ("Columna",             "'rating'  (3ª columna)"),
    ("Valores conocidos",   miles(obs)),
    ("Valores a predecir",  miles(vacias)),
    ("Se optimiza",         "MINIMIZANDO el error de prediccion"),
    ("Metrica",             "RMSE, en estrellas"),
    ("Umbral de exito",     "RMSE <= 0,90     (hoy, por popularidad: 0,977)"),
]
y = .855
for k, v in ficha:
    destaca = k in ("Se optimiza", "Umbral de exito", "Valores a predecir")
    ax.text(.035, y, k, fontsize=10.5, fontweight="bold",
            color="#8b0000" if destaca else "#333333", transform=ax.transAxes)
    ax.text(.40, y, v, fontsize=10.5,
            fontweight="bold" if destaca else "normal",
            color="#8b0000" if destaca else "#000000", transform=ax.transAxes)
    y -= .0785

sal = f"{RAIZ}/01_comprension_negocio/graficos/3_3_variable_a_optimizar.png"
fig.savefig(sal, dpi=150, facecolor="white")
print("  ->", sal)
print("Listo.")
