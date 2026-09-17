"""Imagenette results: replication, collapse control, and our lambda sweep.

    python imagenette/analyze.py                          # reads results/imagenette.json
    python imagenette/analyze.py --runs runs/imagenette   # or a directory of run folders (with probe.json)
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load(runs):
    if runs is None:
        return json.loads((ROOT / "results/imagenette.json").read_text())
    rows = []
    for d in sorted(Path(runs).glob("*/")):
        if not (d / "result.json").exists(): continue
        r = json.loads((d / "result.json").read_text()); pr = json.loads((d / "probe.json").read_text()) if (d / "probe.json").exists() else {}
        rows.append(dict(method=r["method"], lamb=r["lamb"], seed=r["seed"], online_final=r["final_acc"] * 100, online_best=r["best_acc"] * 100,
                         offline_linear=pr.get("offline_linear"), backbone_rank=pr.get("backbone_rank"), epochs=len(r["history"]),
                         terminated_early=len(r["history"]) < r["epochs"]))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs"); a = p.parse_args()
    rows = load(a.runs)
    pct = lambda x: "    n/a" if x is None else f"{x:6.2f}%"
    num = lambda x: "   n/a" if x is None else f"{x:6.1f}"
    print(f"{'method':7s} {'lambda':>6s} {'online':>7s} {'best':>7s} {'offline':>8s} {'rank':>6s}")
    for r in sorted(rows, key=lambda r: (r["method"], r["lamb"])):
        print(f"{r['method']:7s} {r['lamb']:6g} {r['online_final']:6.2f}% {r['online_best']:6.2f}% {pct(r['offline_linear'])} {num(r['backbone_rank'])}"
              + ("  (stopped early)" if r.get("terminated_early") else ""))
    sig = next(r for r in rows if r["method"] == "sigreg")
    epi = sorted([r for r in rows if r["method"] == "epi" and not r.get("terminated_early")], key=lambda r: r["lamb"])
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot([r["lamb"] for r in epi], [r["online_final"] for r in epi], "o-", label="ours, online probe (their metric)")
    off = [r for r in epi if r["offline_linear"] is not None]
    ax.plot([r["lamb"] for r in off], [r["offline_linear"] for r in off], "^-", label="ours, frozen-backbone probe")
    ax.axhline(sig["online_final"], color="tab:green", ls="--", label=f"SIGReg online {sig['online_final']:.1f}")
    if sig["offline_linear"]: ax.axhline(sig["offline_linear"], color="tab:green", ls=":", label=f"SIGReg frozen {sig['offline_linear']:.1f}")
    ax.set_xscale("log"); ax.set_xticks([r["lamb"] for r in epi]); ax.set_xticklabels([f"{r['lamb']:g}" for r in epi])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="lambda", ylabel="Imagenette accuracy (%)", title="Imagenette, ViT-S/8, 800 epochs (LeJEPA chassis)"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); out = ROOT / "results/imagenette.png"; fig.savefig(out, dpi=150); print("\nsaved", out)


if __name__ == "__main__":
    main()
