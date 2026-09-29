"""v21.0-baseline — entrypoint Colab T4. Só telemetria mínima, sem treino pesado ainda."""
import json, csv, pathlib
ver = "v21.0-baseline"
cfg = json.loads(pathlib.Path(f"bundles/{ver}/config.json").read_text())
print(f"[{ver}] config:", cfg)
# TODO: importar ConsciousModel + TANGO aqui; por ora fumaça das regras
out = pathlib.Path(f"bundles/{ver}/metrics.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss_train", "loss_val", "clip_rate", "grad_norm", "gap"])
    w.writerow([0, 0, 0, 0, 0, 0])
pathlib.Path(f"bundles/{ver}/teste_contexto.log").write_text("layer_scale 1.0 vs 0.0: PENDENTE\n")
print(f"[{ver}] fumaça ok -> {out}")
