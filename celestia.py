#!/usr/bin/env python3
"""CLI ponte PC -> Colab (grátis, sem Drive manual).
Fluxo: PC prepara bundle versionado -> tu cola 1 célula no Colab -> Colab executa e empurra resultados p/ GitHub -> PC puxa e valida pelas regras.
Uso:
  python celestia.py new-version v21.0-baseline
  python celestia.py colab-cell v21.0-baseline --repo <url-github> [--branch main]
  # após rodar no Colab:
  python celestia.py check v21.0-baseline
"""
import sys, json, hashlib
from pathlib import Path
ROOT = Path(__file__).parent
BUNDLE = ROOT / "bundles"

BASE_CONFIG = {
  "n_layers": 12, "d_model": 768, "seq_len": 128, "vocab_size": 32000,
  "facts_frac": 0.05, "facts_anchored": True,
  "beta_start": 0.15, "clip_norm": 1.5,
  "seeds": [0, 1, 2], "hardware_ref": "colab-t4-free",
}

def cmd_new(ver):
    d = BUNDLE / ver
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(BASE_CONFIG, version=ver)
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    (d / "RESULTADO.md").write_text(f"# {ver}\nStatus: aguardando Colab\n")
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    (d / "bundle.hash").write_text(h + "\n")
    print(f"ok -> {d} hash={h}")

def cmd_cell(ver, repo, branch="main"):
    h = (BUNDLE / ver / "bundle.hash").read_text().strip() if (BUNDLE/ver/"bundle.hash").exists() else "?"
    cell = f"""# {ver} [{h}] — cola e roda (1 célula só)
!rm -rf /tmp/cel && git clone --depth 1 -b {branch} {repo} /tmp/cel 2>&1 | tail -n 2
%cd /tmp/cel
!pip -q install numpy scipy sympy scikit-learn 2>&1 | tail -n 1
!python run_{ver.replace('.','_').replace('-','_')}.py 2>&1 | tee /tmp/{ver}.log
# resultados leves -> GitHub (pesos ficam no Drive, só hash+csv+log sobem)
!python push_results.py --ver {ver} --hash {h} 2>&1 | tail -n 5
print("FIM {ver} {h} — volta no PC e roda: python celestia.py check {ver}")
"""
    print(cell)

def cmd_check(ver):
    d = BUNDLE / ver
    print(f"== check {ver} ==")
    for f in ["config.json", "bundle.hash", "metrics.csv", "teste_contexto.log", "RESULTADO.md"]:
        print(("OK  " if (d/f).exists() else "FALTA"), f)
    print("Regra: sem metrics.csv + teste_contexto.log = inválido (não entra em ranking).")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    cmd, ver = sys.argv[1], sys.argv[2]
    if cmd == "new-version": cmd_new(ver)
    elif cmd == "colab-cell":
        repo = sys.argv[sys.argv.index("--repo")+1] if "--repo" in sys.argv else "SEU-REPO-GITHUB"
        cmd_cell(ver, repo)
    elif cmd == "check": cmd_check(ver)
    else: print("cmd: new-version | colab-cell | check")
