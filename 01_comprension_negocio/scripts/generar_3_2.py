#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gráficos de apoyo para el punto 3.2 (Tipo de Tarea de Minería de Datos)
de la fase de Comprensión del Negocio — CRISP-DM.

Genera dos imágenes resumidas para la presentación:
  1. Qué tarea es y por qué se descartan las demás.
  2. Qué predice exactamente el modelo (la salida es un número, no una etiqueta).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SALIDA = Path("comprension_negocio")
DATOS = next(Path(".").glob("datos_procesados*/datos_procesados"))

C_SUP, C_TINTA, C_T2, C_SUAVE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
C_EJE, C_GRILLA = "#c3c2b7", "#e1e0d9"
C_SI, C_SI_SUAVE = "#2a78d6", "#cde2fb"       # la tarea que sí aplica
C_NO = "#b9b7b0"                               # las descartadas
C_AUX, C_AUX_SUAVE = "#eb6834", "#fbe0d5"      # la que aparece como apoyo

plt.rcParams.update({
    "figure.facecolor": C_SUP, "axes.facecolor": C_SUP, "savefig.facecolor": C_SUP,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "savefig.dpi": 160, "savefig.bbox": "tight"})


def grafico_tipo_de_tarea():
    """Checklist visual: qué se marca y por qué se descarta el resto."""
    opciones = [
        ("Clasificación Supervisada", "no",
         "Predeciría una etiqueta ('le gusta' / 'no le gusta').\nNosotros predecimos la nota exacta, que da más información."),
        ("Regresión / Predicción Numérica", "si",
         "La salida es un número de 1 a 5 estrellas.\nSe evalúa con RMSE y MAE, métricas de regresión."),
        ("Agrupamiento / Clustering", "aux",
         "Se usa como APOYO: agrupar usuarios parecidos para\nformar vecindarios. Es un medio, no el objetivo."),
        ("Reglas de Asociación", "no",
         "Sería 'quien vio A también vio B'. Solo usa si vio o no\nla película, y desperdicia la nota de 1 a 5."),
        ("Detección de Anomalías", "no",
         "Serviría para detectar votos falsos, pero no es\nel objetivo del proyecto."),
        ("Otra: Sistemas de Recomendación\n(Filtrado Colaborativo)", "si",
         "Es el nombre propio de esta tarea dentro de la minería\nde datos: predecir preferencias en una matriz usuario-ítem."),
    ]

    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(opciones) * 1.32 + 0.5); ax.axis("off")

    for i, (nombre, estado, razon) in enumerate(reversed(opciones)):
        y = i * 1.32 + 0.3
        marcada = estado in ("si", "aux")
        c_texto = C_TINTA if marcada else C_SUAVE
        c_caja = C_SI if estado == "si" else (C_AUX if estado == "aux" else C_NO)
        c_fondo = C_SI_SUAVE if estado == "si" else (C_AUX_SUAVE if estado == "aux" else C_SUP)

        # Fondo resaltado solo en las que aplican: el ojo va directo a ellas.
        if marcada:
            ax.add_patch(FancyBboxPatch((0.15, y - 0.05), 9.7, 1.12,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=c_fondo, edgecolor="none", zorder=1))
        # Casilla de verificación.
        ax.add_patch(plt.Rectangle((0.45, y + 0.34), 0.42, 0.42, facecolor=c_fondo,
                                   edgecolor=c_caja, linewidth=2, zorder=3))
        if marcada:
            ax.plot([0.53, 0.64, 0.81], [y + 0.55, y + 0.42, y + 0.68],
                    color=c_caja, linewidth=3, solid_capstyle="round", zorder=4)

        ax.text(1.15, y + 0.72, nombre, fontsize=12.5, va="center",
                fontweight="bold" if marcada else "normal", color=c_texto, zorder=4)
        ax.text(1.15, y + 0.24, razon, fontsize=9.5, va="center",
                color=C_T2 if marcada else C_SUAVE, linespacing=1.35, zorder=4)

        etiqueta = {"si": "SÍ", "aux": "APOYO", "no": "no"}[estado]
        ax.text(9.6, y + 0.55, etiqueta, fontsize=10.5, ha="right", va="center",
                fontweight="bold", color=c_caja, zorder=4)

    ax.set_title("3.2  Tipo de Tarea de Minería de Datos", fontsize=17,
                 fontweight="bold", color=C_TINTA, loc="left", pad=32)
    ax.text(0, 1.005,
            "Sistema de recomendación de películas por filtrado colaborativo (MovieLens 1M)",
            transform=ax.transAxes, fontsize=11, color=C_T2, va="bottom")
    fig.tight_layout()
    fig.savefig(SALIDA / "3_2_tipo_de_tarea.png")
    plt.close(fig)
    print("  [ok] 3_2_tipo_de_tarea.png")


def grafico_que_predice():
    """
    Hace tangible por qué es regresión: la salida es un número continuo.
    Panel izquierdo: la celda vacía que hay que rellenar.
    Panel derecho: la variable objetivo real, con su rango y su media.
    """
    rep = json.loads((DATOS / "reporte_preparacion.json").read_text(encoding="utf-8"))
    dist = rep["inspeccion"]["distribucion"]
    media = rep["inspeccion"]["rating_media"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- Panel izquierdo: la pregunta que responde el modelo ---------------
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 7.2); ax1.axis("off")
    rng = np.random.default_rng(7)
    pelis = ["Toy Story", "Star Wars", "Titanic", "Matrix", "Fargo"]
    usuarios = ["Ana", "Luis", "Eva", "Tú"]
    celdas = rng.choice([1, 2, 3, 4, 5, 0, 0], size=(4, 5))
    celdas[3, 3] = -1   # la celda que se quiere predecir

    x0, y0, w, h = 1.9, 1.4, 1.34, 0.92
    for i in range(4):
        for j in range(5):
            v = celdas[i, j]
            x, y = x0 + j * w, y0 + (3 - i) * h
            if v == -1:
                ax1.add_patch(plt.Rectangle((x, y), w - 0.1, h - 0.08, facecolor="#fff3ec",
                                            edgecolor=C_AUX, linewidth=2.4, zorder=3))
                ax1.text(x + (w - 0.1) / 2, y + (h - 0.08) / 2, "?", ha="center", va="center",
                         fontsize=21, fontweight="bold", color=C_AUX, zorder=4)
            elif v == 0:
                ax1.add_patch(plt.Rectangle((x, y), w - 0.1, h - 0.08, facecolor="#f2f1ec",
                                            edgecolor="none", zorder=3))
            else:
                ax1.add_patch(plt.Rectangle((x, y), w - 0.1, h - 0.08,
                                            facecolor=C_SI_SUAVE, edgecolor="none", zorder=3))
                ax1.text(x + (w - 0.1) / 2, y + (h - 0.08) / 2, str(v), ha="center",
                         va="center", fontsize=13, color=C_TINTA, zorder=4)
        ax1.text(x0 - 0.2, y0 + (3 - i) * h + h / 2, usuarios[i], ha="right", va="center",
                 fontsize=11, color=C_TINTA if i == 3 else C_T2,
                 fontweight="bold" if i == 3 else "normal")
    for j, p in enumerate(pelis):
        ax1.text(x0 + j * w + (w - 0.1) / 2, y0 + 4 * h + 0.12, p, ha="center",
                 va="bottom", fontsize=9.5, color=C_T2, rotation=28)

    ax1.annotate("", xy=(9.3, 1.86), xytext=(8.8, 1.86),
                 arrowprops=dict(arrowstyle="-|>", color=C_AUX, linewidth=2.2))
    ax1.text(9.45, 1.86, "4,2 ★", fontsize=17, fontweight="bold", color=C_AUX, va="center")
    ax1.text(5.0, 0.55, "El modelo rellena los huecos: dice qué nota daría cada usuario\n"
             "a una película que todavía no ha visto.",
             ha="center", fontsize=10.5, color=C_T2, linespacing=1.4)
    ax1.set_title("Qué predice el modelo", fontsize=13.5, fontweight="bold",
                  color=C_TINTA, loc="left", pad=14)

    # ---- Panel derecho: la variable objetivo es numérica -------------------
    valores = [float(k) for k in dist]
    conteos = [dist[k] for k in dist]
    total = sum(conteos)
    ax2.bar(valores, conteos, width=0.66, color=C_SI, zorder=3, linewidth=0)
    for v, c in zip(valores, conteos):
        ax2.text(v, c, f"{100*c/total:.0f}%", ha="center", va="bottom",
                 fontsize=9.5, color=C_T2)
    ax2.axvline(media, color=C_AUX, linestyle="--", linewidth=1.8, zorder=4)
    ax2.text(media - 0.08, ax2.get_ylim()[1] * 0.98, f"media {media:.2f} ",
             ha="right", va="top", fontsize=10.5, fontweight="bold", color=C_AUX)

    ax2.set_title("La variable objetivo es un número", fontsize=13.5,
                  fontweight="bold", color=C_TINTA, loc="left", pad=14)
    ax2.set_xlabel("Calificación (1 a 5 estrellas)", fontsize=10.5, color=C_T2)
    ax2.set_ylabel("Nº de calificaciones", fontsize=10.5, color=C_T2)
    ax2.set_xticks(valores)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{x/1000:.0f}k" if x else "0"))
    for lado in ["top", "right"]:
        ax2.spines[lado].set_visible(False)
    ax2.spines["left"].set_color(C_EJE); ax2.spines["bottom"].set_color(C_EJE)
    ax2.tick_params(colors=C_SUAVE, labelsize=9)
    ax2.grid(axis="y", color=C_GRILLA, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle("Por qué es Regresión y no Clasificación", fontsize=16,
                 fontweight="bold", color=C_TINTA, y=1.10)
    fig.text(0.5, 1.035,
             "Se predice un valor continuo (4,2 estrellas), no una etiqueta ('le gusta' / 'no le gusta')",
             ha="center", fontsize=10.5, color=C_T2)
    fig.tight_layout()
    fig.savefig(SALIDA / "3_2_por_que_regresion.png")
    plt.close(fig)
    print("  [ok] 3_2_por_que_regresion.png")


if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    print(f"Datos leídos de: {DATOS}")
    grafico_tipo_de_tarea()
    grafico_que_predice()
    print("Listo.")
