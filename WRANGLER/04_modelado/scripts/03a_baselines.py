# -*- coding: utf-8 -*-
"""
ESCALERA DE BASELINES  -  fase 4 (modelado)
Calcula los 4 modelos de referencia SIN librerias externas (solo pandas/numpy),
con hold-out 80/20 y SIN fuga de datos: todo se estima solo con entrenamiento.

Uso: ./.venv/bin/python 04_modelado/scripts/03a_baselines.py
"""
# --- funciona desde cualquier carpeta: se situa en la raiz del proyecto ---
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, json, time

SEED, LAMBDA = 42, 2.0          # lambda = 2 es el minimo medido en el barrido
LO, HI = 1.0, 5.0

df = pd.read_parquet("03_preparacion_datos/datos/datos_limpios.parquet")[
        ["userId","movieId","rating"]]
print(f"Cargado: {len(df):,} calificaciones")

# ---------- SPLIT 80/20 (el test NO se toca nunca al estimar) ----------
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(df))
corte = int(len(df)*0.8)
tr = df.iloc[idx[:corte]].reset_index(drop=True)
te = df.iloc[idx[corte:]].reset_index(drop=True)
print(f"Entrenamiento: {len(tr):,}   Prueba: {len(te):,}\n")

def metricas(pred, real):
    pred = np.clip(pred, LO, HI)          # el rating no puede salirse de [1,5]
    e = real - pred
    return float(np.sqrt((e**2).mean())), float(np.abs(e).mean())

real = te.rating.to_numpy(float)
res  = {}

# ---------- ESCALON 1: MEDIA GLOBAL ----------
mu = tr.rating.mean()
res["1. Media global"] = metricas(np.full(len(te), mu), real) + (f"mu = {mu:.4f}",)

# ---------- ESCALON 2: MEDIA POR USUARIO ----------
mu_u = tr.groupby("userId").rating.mean()
p = te.userId.map(mu_u).fillna(mu).to_numpy(float)     # usuario nuevo -> mu
res["2. Media por usuario"] = metricas(p, real) + (f"{len(mu_u):,} medias",)

# ---------- ESCALON 3: MEDIA POR PELICULA (= popularidad, practica actual) ----------
mu_i = tr.groupby("movieId").rating.mean()
p = te.movieId.map(mu_i).fillna(mu).to_numpy(float)    # pelicula nueva -> mu
res["3. Media por pelicula"] = metricas(p, real) + (f"{len(mu_i):,} medias",)

# ---------- ESCALON 4: MEDIA + SESGOS  (mu + bu + bi, regularizados) ----------
tr = tr.assign(dev=tr.rating - mu)
g = tr.groupby("movieId").dev
bi = (g.sum() / (LAMBDA + g.size()))                   # sesgo de pelicula
tr = tr.assign(dev2=tr.dev - tr.movieId.map(bi))
g = tr.groupby("userId").dev2
bu = (g.sum() / (LAMBDA + g.size()))                   # sesgo de usuario
p = (mu + te.userId.map(bu).fillna(0) + te.movieId.map(bi).fillna(0)).to_numpy(float)
res["4. Media + sesgos"] = metricas(p, real) + (f"lambda = {LAMBDA:.0f}",)

# ---------- SALIDA ----------
print(f"{'MODELO':<26}{'RMSE':>9}{'MAE':>9}   detalle")
print("-"*62)
for k,(r,m,d) in res.items():
    print(f"{k:<26}{r:>9.4f}{m:>9.4f}   {d}")

pop = res["3. Media por pelicula"][0]
print("\nMEJORA DE CADA ESCALON FRENTE A POPULARIDAD (RMSE 0,977 = practica actual):")
for k,(r,m,d) in res.items():
    print(f"  {k:<26} {(pop-r)/pop*100:+7.2f}%")

json.dump({k: {"rmse": r, "mae": m, "detalle": d} for k,(r,m,d) in res.items()} |
          {"_config": {"seed": SEED, "lambda": LAMBDA,
                       "n_train": len(tr), "n_test": len(te)},
           "generado": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
          open("04_modelado/resultados/baselines.json","w"), indent=2)
print("\n-> 04_modelado/resultados/baselines.json")
