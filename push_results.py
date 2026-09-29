"""Sobe só leves de volta (o Colab roda isso no fim). Pesos .h5 ficam no Drive."""
import argparse, subprocess, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--ver", required=True)
ap.add_argument("--hash", required=True)
a = ap.parse_args()
d = pathlib.Path(f"bundles/{a.ver}")
print(f"[{a.ver} {a.hash}] arquivos:", [p.name for p in d.glob('*')])
# no Colab com token GH, descomentar:
# subprocess.run(["git", "add", f"bundles/{a.ver}/metrics.csv",
#                 f"bundles/{a.ver}/teste_contexto.log"], check=False)
# subprocess.run(["git", "commit", "-m", f"{a.ver} {a.hash} resultados"], check=False)
# subprocess.run(["git", "push"], check=False)
print("push_results: conecta teu token depois; por ora manual via Drive/GitHub web.")
