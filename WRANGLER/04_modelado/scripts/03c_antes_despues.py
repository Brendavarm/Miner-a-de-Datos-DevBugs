# -*- coding: utf-8 -*-
"""ANTES vs DESPUES: practica actual (popularidad) frente al modelo entrenado."""
# --- funciona desde cualquier carpeta: se situa en la raiz del proyecto ---
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, json

d = np.load("04_modelado/resultados/modelo_mf.npz", allow_pickle=True)
P,Q,bu,bi,mu = d["P"],d["Q"],d["bu"],d["bi"],float(d["mu"])
uidx = {u:k for k,u in enumerate(d["users"])}; iidx = {i:k for k,i in enumerate(d["items"])}

df = pd.read_parquet("03_preparacion_datos/datos/datos_limpios.parquet")
mv = pd.read_csv("03_preparacion_datos/datos/mapa_peliculas.csv").set_index("movieId")
rng = np.random.default_rng(42); idx = rng.permutation(len(df)); c = int(len(df)*.8)
tr, te = df.iloc[idx[:c]], df.iloc[idx[c:]]
nvot = tr.groupby("movieId").size(); mi = tr.groupby("movieId").rating.mean()

def pred(u, items):
    ui = uidx[u]; ii = np.array([iidx[i] for i in items])
    p = mu + bu[ui] + bi[ii] + P[ui] @ Q[ii].T
    return np.clip(p, 1, 5)

# ═══ 1) AGREGADO ═══
cnt = te.groupby("userId").size(); ok = set(cnt[cnt>=10].index)
sub = te[te.userId.isin(ok)]
pr = []
for u, g in sub.groupby("userId"):
    it = g.movieId.to_numpy(); real = g.rating.to_numpy()
    top_m = np.argsort(-pred(u, it))[:10]
    top_p = np.argsort(-np.nan_to_num(it_med := g.movieId.map(mi.where(nvot>=50)).to_numpy(float)))[:10]
    pr.append(((real[top_m]>=4).mean(), (real[top_p]>=4).mean()))
pr = np.array(pr)
print("="*66)
print("ANTES vs DESPUES  ·  agregado sobre", f"{len(pr):,}", "usuarios")
print("="*66)
print(f"{'':28}{'ANTES':>12}{'DESPUES':>12}{'mejora':>12}")
print(f"{'RMSE (estrellas)':28}{0.9765:>12.4f}{0.8456:>12.4f}{'+13,4%':>12}")
print(f"{'MAE (estrellas)':28}{0.7800:>12.4f}{0.6648:>12.4f}{'+14,8%':>12}")
print(f"{'Precision@10':28}{pr[:,1].mean()*100:>11.2f}%{pr[:,0].mean()*100:>11.2f}%"
      f"{(pr[:,0].mean()-pr[:,1].mean())*100:>+11.1f} pp")

# ═══ 2) CASO DE UN USUARIO ═══
U = 1150
g = te[te.userId==U]
it = g.movieId.to_numpy(); real = g.rating.to_numpy()
pm = pred(U, it)
pp = g.movieId.map(mi.where(nvot>=50)).to_numpy(float)
print("\n"+"="*66)
print(f"CASO REAL  ·  usuario {U}  (media personal {tr[tr.userId==U].rating.mean():.2f}"
      f" frente a {mu:.2f} global)")
print("="*66)
for etq, orden, val in [("ANTES  — Top-10 por popularidad", np.argsort(-np.nan_to_num(pp)), pp),
                        ("DESPUES— Top-10 del modelo",      np.argsort(-pm),                pm)]:
    t = orden[:10]
    print(f"\n{etq}   ->  acierto {int((real[t]>=4).sum())}/10")
    for r_ in t:
        ac = "OK " if real[r_]>=4 else "  X"
        print(f"   {ac} {mv.loc[it[r_],'title'][:44]:<46} predice {val[r_]:.2f}  real {real[r_]:.0f}")

json.dump({"p10_modelo":float(pr[:,0].mean()),"p10_popularidad":float(pr[:,1].mean()),
           "n_usuarios":int(len(pr)),
           "generado": __import__("datetime").datetime.now().isoformat(timespec="seconds")}, open("04_modelado/resultados/antes_despues.json","w"), indent=2)
