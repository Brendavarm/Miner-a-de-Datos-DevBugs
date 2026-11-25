# -*- coding: utf-8 -*-
"""
Genera el material de apoyo del punto 3.3:
la VARIABLE A OPTIMIZAR r(u,i) = columna 'rating'.

Salidas:
  WRANGLER/01_comprension_negocio/graficos/3_3_variable_a_optimizar.png

Uso:  python WRANGLER/01_comprension_negocio/scripts/generar_3_3.py
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

def miles(n):
    """Separador de miles español."""
    return f"{n:,}".replace(",", ".")

RAIZ = Path(__file__).resolve().parents[3]
DATOS = RAIZ / "anexo_ml-latest-small/ml-latest-small"
RATINGS = DATOS / "ratings.csv"
SALIDA = RAIZ / "WRANGLER/01_comprension_negocio/graficos/3_3_variable_a_optimizar.png"

print("Cargando", RATINGS, "...")
df = pd.read_csv(RATINGS).merge(pd.read_csv(DATOS / "movies.csv"), on="movieId", how="left")
print(f"  {len(df):,} filas, columnas: {list(df.columns)}")

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
         f"MovieLens ml-latest-small  ·  {miles(obs)} calificaciones  ·  {miles(n_users)} usuarios  ·  {miles(n_items)} peliculas calificadas",
         ha="center", fontsize=11, color="#555555")

AZUL, ROJO, GRIS = "#1f77b4", "#d62728", "#888888"

# --- (A) forma larga: la tabla ---------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
ax.set_title("A)  Donde esta:  columna 'rating' de ratings.csv",
             fontsize=12, fontweight="bold", loc="left", pad=14)

filas = df[["userId", "movieId", "rating", "title"]].head(6).values
cab = ["userId\n(u)", "movieId\n(i)", "rating\nr(u,i)", "title"]
tabla = ax.table(
    cellText=[[f"{int(a)}", f"{int(b)}", f"{c:g}", (d[:26] + "…") if len(d) > 26 else d]
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
        f"Cada fila = una celda YA OBSERVADA.   Ejemplo:  r({int(filas[0][0])}, {int(filas[0][1])}) = {filas[0][2]:g}",
        ha="center", fontsize=11, style="italic", color="#333333",
        transform=ax.transAxes)
ax.text(0.5, 0.04, f"{miles(obs)} filas como estas  ({obs/total*100:.2f}% de la matriz)",
        ha="center", fontsize=10, color=GRIS, transform=ax.transAxes)

# --- (B) forma matriz: los huecos ------------------------------------
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.set_title("B)  Que hay que optimizar:  celdas vacias (esquema)",
             fontsize=13, fontweight="bold", loc="left", pad=14)

rng = np.random.default_rng(7)
nf, nc = 6, 8
ax.set_xlim(-1.55, nc + 0.4); ax.set_ylim(-1.5, nf + 1.75); ax.invert_yaxis()
for i in range(nf):
    for j in range(nc):
        hay = (i == 0 and j == 0) or rng.random() < 0.017
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
ax.text(-1.35, nf/2, f"usuarios\n{miles(n_users)}", rotation=90, ha="center", va="center",
        fontsize=11, fontweight="bold")
ax.text(nc/2, -1.05, f"peliculas   {miles(n_items)}", ha="center", fontsize=11, fontweight="bold")

ax.add_patch(Rectangle((0, nf + .35), .8, .55, facecolor="#cfe3f3", edgecolor="#b0b0b0"))
ax.text(1.0, nf + .63, f"observadas: {miles(obs)}  ({obs/total*100:.2f}%)  → datos disponibles",
        va="center", fontsize=10.5)
ax.add_patch(Rectangle((0, nf + 1.0), .8, .55, facecolor="#fdecec", edgecolor="#b0b0b0"))
ax.text(1.0, nf + 1.28, f"vacias: {miles(vacias)}  ({vacias/total*100:.2f}%)  → PREDECIR",
        va="center", fontsize=10.5, fontweight="bold", color=ROJO)

# --- (C) distribucion de la variable ---------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.set_title("C)  Como se distribuye la variable objetivo (lo observado)",
             fontsize=13, fontweight="bold", loc="left", pad=12)
barras = ax.bar([f"{k:g}★" for k in dist.index], dist.values,
                color=["#c94b4b", "#d46950", "#e08a5a", "#e9a77e", "#d9c15a", "#b3bf63", "#96b767", "#7fb069", "#63a765", "#4a9d5f"],
                edgecolor="white", lw=1.5)
for b, v in zip(barras, dist.values):
    ax.text(b.get_x() + b.get_width()/2, v + dist.max() * .02,
            f"{v/obs*100:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax.axhline(0, color="#333333", lw=1)
ax.set_ylabel("nº de calificaciones", fontsize=11)
ax.set_ylim(0, dist.max() * 1.55)
ax.spines[["top", "right"]].set_visible(False)
ax.text(.99, .93, f"media = {media:.2f} ★\nescala 0,5 a 5, medias estrellas",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#bbbbbb"))
ax.text(.02, .93, f"{100*(df.rating >= 4).mean():.1f}% son 4★ o mas\n3,5★ no cuenta como Me gusta",
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
    ("Rango",               "[0,5, 5] estrellas"),
    ("Fichero",             "anexo_ml-latest-small/ml-latest-small/ratings.csv"),
    ("Columna",             "'rating'  (3ª columna)"),
    ("Valores conocidos",   miles(obs)),
    ("Valores a predecir",  miles(vacias)),
    ("Se optimiza",         "MINIMIZANDO el error de prediccion"),
    ("Metrica",             "RMSE, en estrellas"),
    ("Umbral de exito",     "RMSE <= 0,90     (media por pelicula: 0,973)"),
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

fig.savefig(SALIDA, dpi=150, facecolor="white")
print("  ->", SALIDA)
print("Listo.")
