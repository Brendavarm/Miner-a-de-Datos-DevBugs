# -*- coding: utf-8 -*-
"""
FACTORIZACION MATRICIAL (estilo FunkSVD) implementada en numpy puro.
No necesita scikit-surprise. Sirve para tener un "despues" REAL medido por
nosotros mientras no se pueda instalar la libreria.

  prediccion:  r(u,i) = mu + b_u + b_i + p_u . q_i
  entrenamiento: descenso de gradiente por mini-lotes sobre el 80%

SIN fuga de datos: la epoca en la que se para se elige con una VALIDACION
(10% del entrenamiento). El test solo se usa una vez, al final.

  1) se entrena con el 90% del train y se mide la validacion en cada epoca
  2) parada temprana: la mejor epoca es el minimo de la validacion
  3) se reentrena desde cero con TODO el train durante esas epocas
  4) se mide el test una sola vez

Uso: ./.venv/bin/python 04_modelado/scripts/03b_mf_numpy.py
"""
# --- funciona desde cualquier carpeta: se situa en la raiz del proyecto ---
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, json, time

# Valores por defecto. Se pueden cambiar al vuelo, por ejemplo:
#   ./.venv/bin/python 04_modelado/scripts/03b_mf_numpy.py --k 128 --reg 0.02
import sys
def _arg(nombre, defecto, tipo=float):
    etq = "--" + nombre
    return tipo(sys.argv[sys.argv.index(etq)+1]) if etq in sys.argv else defecto

SEED   = _arg("seed",   42,    int)
K      = _arg("k",      64,    int)     # nº de factores latentes
LR     = _arg("lr",     0.010)          # tasa de aprendizaje
# k=64/lr=0.010 se eligio por la VALIDACION: RMSE 0,8549 frente a 0,8581 de k=32/lr=0.020
REG    = _arg("reg",    0.045)          # regularizacion
EPOCHS = _arg("epochs", 45,    int)
BATCH  = _arg("batch",  4096,  int)
PACIENCIA = 6   # epocas sin mejorar antes de parar
VALIDACION = 0.10   # parte del train que se aparta para decidir cuando parar
LO, HI = 1.0, 5.0

df = pd.read_parquet("03_preparacion_datos/datos/datos_limpios.parquet")[
        ["userId","movieId","rating"]]

# ---- indices densos ----
uc = df.userId.astype("category"); ic = df.movieId.astype("category")
U = uc.cat.codes.to_numpy(np.int32); I = ic.cat.codes.to_numpy(np.int32)
R = df.rating.to_numpy(np.float32)
nU, nI = len(uc.cat.categories), len(ic.cat.categories)

# ---- split 80/20 identico al de los baselines ----
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(df)); cut = int(len(df)*0.8)
tr, te = idx[:cut], idx[cut:]
# ---- validacion: el ultimo 10% del train (ya esta barajado) ----
cut_val = int(len(tr)*(1-VALIDACION))
aj, va = tr[:cut_val], tr[cut_val:]
print(f"usuarios={nU}  peliculas={nI}  train={len(tr):,} (ajuste={len(aj):,} + validacion={len(va):,})"
      f"  test={len(te):,}")

def iniciar():
    r2 = np.random.default_rng(7)
    P  = r2.normal(0, 0.05, (nU, K)).astype(np.float32)
    Q  = r2.normal(0, 0.05, (nI, K)).astype(np.float32)
    return r2, P, Q, np.zeros(nU, np.float32), np.zeros(nI, np.float32)

def una_epoca(filas, mu, r2, P, Q, bu, bi):
    orden = filas.copy(); r2.shuffle(orden)
    for s in range(0, len(orden), BATCH):
        b = orden[s:s+BATCH]
        u, i, r = U[b], I[b], R[b]
        pu, qi = P[u], Q[i]
        err = r - (mu + bu[u] + bi[i] + np.einsum("ij,ij->i", pu, qi))
        e = err[:, None]
        np.add.at(bu, u, LR*(err - REG*bu[u]))
        np.add.at(bi, i, LR*(err - REG*bi[i]))
        np.add.at(P, u, LR*(e*qi - REG*pu))
        np.add.at(Q, i, LR*(e*pu - REG*qi))

def evaluar(filas, mu, P, Q, bu, bi):
    p = mu + bu[U[filas]] + bi[I[filas]] + np.einsum("ij,ij->i", P[U[filas]], Q[I[filas]])
    p = np.clip(p, LO, HI); e = R[filas] - p
    return float(np.sqrt((e**2).mean())), float(np.abs(e).mean())

# ═══ 1) y 2) ELEGIR LA EPOCA CON LA VALIDACION ═══
print(f"\n{'epoca':>6}{'RMSE valid':>12}{'MAE valid':>11}")
print("-"*29)
t0 = time.time(); hist = []
mu = R[aj].mean(); r2, P, Q, bu, bi = iniciar()
mejor = (9e9, 9e9, 0); sin_mejora = 0
for ep in range(1, EPOCHS+1):
    una_epoca(aj, mu, r2, P, Q, bu, bi)
    rm, ma = evaluar(va, mu, P, Q, bu, bi); hist.append((ep, rm, ma))
    if ep % 3 == 0 or ep == 1:
        marca = "  <- mejor" if rm <= min(h[1] for h in hist) else ""
        print(f"{ep:>6}{rm:>12.4f}{ma:>11.4f}{marca}")
    # PARADA TEMPRANA: recuerda la mejor epoca y corta si deja de mejorar
    if rm < mejor[0]:
        mejor = (rm, ma, ep); sin_mejora = 0
    else:
        sin_mejora += 1
        if sin_mejora >= PACIENCIA:
            print(f"\n  Parada temprana en la epoca {ep}: {PACIENCIA} epocas sin mejorar.")
            break
rmse_val, mae_val, mejor_ep = mejor
print(f"  Mejor epoca segun la validacion: {mejor_ep}  (RMSE valid {rmse_val:.4f})")

# ═══ 3) REENTRENAR CON TODO EL TRAIN ═══
mu = R[tr].mean(); r2, P, Q, bu, bi = iniciar()
for ep in range(mejor_ep):
    una_epoca(tr, mu, r2, P, Q, bu, bi)

# ═══ 4) TEST: se mide una sola vez ═══
rmse, mae = evaluar(te, mu, P, Q, bu, bi)
bl = json.load(open("04_modelado/resultados/baselines.json"))
pop, ses = bl["3. Media por pelicula"]["rmse"], bl["4. Media + sesgos"]["rmse"]
print(f"\nEntrenado en {time.time()-t0:.1f} s")
print(f"RESULTADO FINAL (test)   RMSE {rmse:.4f}   MAE {mae:.4f}")
print(f"  vs popularidad ({pop:.4f}): {(pop-rmse)/pop*100:+.2f}%")
print(f"  vs sesgos      ({ses:.4f}): {(ses-rmse)/ses*100:+.2f}%")

np.savez("04_modelado/resultados/modelo_mf.npz", P=P, Q=Q, bu=bu, bi=bi, mu=mu,
         users=np.asarray(uc.cat.categories), items=np.asarray(ic.cat.categories))
json.dump({"rmse":rmse,"mae":mae,"mejor_epoca":mejor_ep,"k":K,"lr":LR,"reg":REG,"epochs":EPOCHS,
           "seed":SEED,"rmse_validacion":rmse_val,"mae_validacion":mae_val,
           "historial":hist,   # (epoca, RMSE, MAE) sobre la VALIDACION, no sobre el test
           "generado": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
          open("04_modelado/resultados/mf_numpy.json","w"), indent=2)
print("-> 04_modelado/resultados/modelo_mf.npz  +  mf_numpy.json")
