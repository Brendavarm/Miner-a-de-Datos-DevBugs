# -*- coding: utf-8 -*-
"""
FACTORIZACION MATRICIAL (estilo FunkSVD) implementada en numpy puro.
No necesita scikit-surprise. Sirve para tener un "despues" REAL medido por
nosotros mientras no se pueda instalar la libreria.

  prediccion:  r(u,i) = mu + b_u + b_i + p_u . q_i
  entrenamiento: descenso de gradiente por mini-lotes sobre el 80%

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
REG    = _arg("reg",    0.045)          # regularizacion
EPOCHS = _arg("epochs", 45,    int)
BATCH  = _arg("batch",  4096,  int)
PACIENCIA = 6   # epocas sin mejorar antes de parar
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
Utr, Itr, Rtr = U[tr], I[tr], R[tr]
Ute, Ite, Rte = U[te], I[te], R[te]
mu = Rtr.mean()
print(f"usuarios={nU}  peliculas={nI}  train={len(tr):,}  test={len(te):,}  mu={mu:.4f}")

# ---- parametros ----
r2 = np.random.default_rng(7)
P  = r2.normal(0, 0.05, (nU, K)).astype(np.float32)
Q  = r2.normal(0, 0.05, (nI, K)).astype(np.float32)
bu = np.zeros(nU, np.float32); bi = np.zeros(nI, np.float32)

def evaluar(Ux, Ix, Rx):
    p = mu + bu[Ux] + bi[Ix] + np.einsum("ij,ij->i", P[Ux], Q[Ix])
    p = np.clip(p, LO, HI); e = Rx - p
    return float(np.sqrt((e**2).mean())), float(np.abs(e).mean())

print(f"\n{'epoca':>6}{'RMSE test':>12}{'MAE test':>11}")
print("-"*29)
t0 = time.time(); hist = []
mejor = (9e9, 9e9, 0, None, None, None, None); sin_mejora = 0
orden = np.arange(len(tr))
for ep in range(1, EPOCHS+1):
    r2.shuffle(orden)
    for s in range(0, len(orden), BATCH):
        b = orden[s:s+BATCH]
        u, i, r = Utr[b], Itr[b], Rtr[b]
        pu, qi = P[u], Q[i]
        err = r - (mu + bu[u] + bi[i] + np.einsum("ij,ij->i", pu, qi))
        e = err[:, None]
        np.add.at(bu, u, LR*(err - REG*bu[u]))
        np.add.at(bi, i, LR*(err - REG*bi[i]))
        np.add.at(P, u, LR*(e*qi - REG*pu))
        np.add.at(Q, i, LR*(e*pu - REG*qi))
    rm, ma = evaluar(Ute, Ite, Rte); hist.append((ep, rm, ma))
    if ep % 3 == 0 or ep == 1:
        marca = "  <- mejor" if rm <= min(h[1] for h in hist) else ""
        print(f"{ep:>6}{rm:>12.4f}{ma:>11.4f}{marca}")
    # PARADA TEMPRANA: guarda la mejor foto y corta si deja de mejorar
    if rm < mejor[0]:
        mejor = (rm, ma, ep, P.copy(), Q.copy(), bu.copy(), bi.copy()); sin_mejora = 0
    else:
        sin_mejora += 1
        if sin_mejora >= PACIENCIA:
            print(f"\n  Parada temprana en la epoca {ep}: {PACIENCIA} epocas sin mejorar.")
            break

# restaurar la mejor foto
rmse, mae, mejor_ep, P, Q, bu, bi = mejor
print(f"  Se restaura la epoca {mejor_ep}, que es el minimo de la curva.")
print(f"\nEntrenado en {time.time()-t0:.1f} s")
print(f"RESULTADO FINAL   RMSE {rmse:.4f}   MAE {mae:.4f}")
print(f"  vs popularidad (0,9765): {(0.9765-rmse)/0.9765*100:+.2f}%")
print(f"  vs sesgos      (0,9051): {(0.9051-rmse)/0.9051*100:+.2f}%")

np.savez("04_modelado/resultados/modelo_mf.npz", P=P, Q=Q, bu=bu, bi=bi, mu=mu,
         users=np.asarray(uc.cat.categories), items=np.asarray(ic.cat.categories))
json.dump({"rmse":rmse,"mae":mae,"mejor_epoca":mejor_ep,"k":K,"lr":LR,"reg":REG,"epochs":EPOCHS,
           "seed":SEED,"historial":hist,
           "generado": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
          open("04_modelado/resultados/mf_numpy.json","w"), indent=2)
print("-> 04_modelado/resultados/modelo_mf.npz  +  mf_numpy.json")
