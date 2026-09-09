# -*- coding: utf-8 -*-
"""
Ejecuta la fase 4 COMPLETA y en el orden correcto:

    03a_baselines.py   ->  baselines.json
    03b_mf_numpy.py    ->  mf_numpy.json + modelo_mf.npz
    03c_antes_despues.py -> antes_despues.json

Usa SIEMPRE este en vez de lanzarlos sueltos: si solo corres 03b, la
Precision@10 del panel se queda con el modelo anterior y los numeros
no cuadran entre si.

Uso:  ./.venv/bin/python 04_modelado/scripts/00_entrenar_todo.py
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AQUI = "04_modelado/scripts"
PASOS = [("03a_baselines.py",     "Modelos de referencia (escalera de baselines)"),
         ("03b_mf_numpy.py",      "Entrenamiento del modelo (factorizacion matricial)"),
         ("03c_antes_despues.py", "Comparativa antes / despues y caso de usuario")]

t0 = time.time()
for i, (script, desc) in enumerate(PASOS, 1):
    print("\n" + "="*70)
    print(f"  PASO {i}/{len(PASOS)}  ·  {desc}")
    print("="*70)
    r = subprocess.run([sys.executable, os.path.join(AQUI, script)])
    if r.returncode != 0:
        print(f"\n!! FALLO en {script} (codigo {r.returncode}). Se detiene aqui.")
        sys.exit(r.returncode)

print("\n" + "="*70)
print(f"  LISTO en {time.time()-t0:.0f} s. Los tres JSON estan sincronizados.")
print("  Recarga 04_modelado/resultados.html en el navegador.")
print("="*70)
