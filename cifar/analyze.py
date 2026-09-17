"""CIFAR-10 results: beta sweep on validation, selected arms on the official test split.

    python cifar/analyze.py                       # reads results/cifar_*.json
    python cifar/analyze.py --runs runs/cifar     # or a fresh run directory
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_results(runs):
    if runs is None:
        return (json.loads((ROOT / "results/cifar_validation.json").read_text()),
                json.loads((ROOT / "results/cifar_test.json").read_text()))
    val = []
    for r in json.loads((Path(runs) / "validation_results.json").read_text()):
        m = r["metrics"]
        val.append(dict(method=r["method"], beta=r["beta"], seed=r["seed"], backbone_val=m["backbone"]["validation_accuracy"] * 100,
                        embedding_val=m["embedding"]["validation_accuracy"] * 100, embedding_rank=m["embedding"]["effective_rank"]))
    test = []
    for r in json.loads((Path(runs) / "test_results.json").read_text()):
        m = r["metrics"]
        test.append(dict(method=r["method"], beta=r["beta"], seed=r["seed"], backbone_test=m["backbone"]["test_accuracy"],
                         embedding_test=m["embedding"]["test_accuracy"], embedding_rank=m["embedding"]["effective_rank"]))
    return val, test


def mean_sd(xs):
    return np.mean(xs), (np.std(xs, ddof=1) if len(xs) > 1 else 0.)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs"); a = p.parse_args()
    val, test = load_results(a.runs)
    print("validation backbone accuracy (mean over seeds):")
    grid = {}
    for r in val:
        grid.setdefault((r["method"], r["beta"]), []).append(r["backbone_val"])
    for (m, b), xs in sorted(grid.items()):
        mu, sd = mean_sd(xs); print(f"  {m:7s} beta {b:5g}: {mu:6.2f} +- {sd:4.2f}   rank {np.mean([r['embedding_rank'] for r in val if (r['method'], r['beta']) == (m, b)]):5.1f}")
    print("\nofficial test split, selected beta per method:")
    for m in ["align", "epi", "sigreg"]:
        rows = [r for r in test if r["method"] == m]
        if not rows: continue
        bt = [r["backbone_test"] * 100 for r in rows]; et = [r["embedding_test"] * 100 for r in rows]
        print(f"  {m:7s} beta {rows[0]['beta']:5g}: backbone {mean_sd(bt)[0]:6.2f} +- {mean_sd(bt)[1]:4.2f}   embedding {mean_sd(et)[0]:6.2f}   rank {np.mean([r['embedding_rank'] for r in rows]):5.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    epi = sorted({b for (m, b) in grid if m == "epi"})
    mu = [mean_sd(grid[("epi", b)])[0] for b in epi]; sd = [mean_sd(grid[("epi", b)])[1] for b in epi]
    ax[0].errorbar(epi, mu, yerr=sd, fmt="o-", capsize=3, label="epiplexity (ours)")
    for m, c, ls in [("sigreg", "tab:green", "--"), ("align", "tab:gray", ":")]:
        xs = [x for (mm, b), x in grid.items() if mm == m]
        best = max((mean_sd(grid[(m, b)])[0] for (mm, b) in grid if mm == m), default=None)
        if best is not None: ax[0].axhline(best, color=c, ls=ls, label=f"{m} (best beta)")
    ax[0].set_xscale("log"); ax[0].set(xlabel="beta", ylabel="validation backbone acc (%)", title="CIFAR-10: weight sweep"); ax[0].legend(); ax[0].grid(alpha=.2)
    names = ["align", "epi", "sigreg"]; vals = [mean_sd([r["backbone_test"] * 100 for r in test if r["method"] == m]) for m in names]
    ax[1].bar(names, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=4, color=["tab:gray", "tab:blue", "tab:green"])
    for i, v in enumerate(vals): ax[1].text(i, v[0] + 0.8, f"{v[0]:.2f}", ha="center")
    ax[1].set(ylabel="test backbone acc (%)", title="CIFAR-10: official test, 3 seeds", ylim=(60, 82)); ax[1].grid(alpha=.2, axis="y")
    fig.tight_layout(); out = ROOT / "results/cifar.png"; fig.savefig(out, dpi=150); print("\nsaved", out)


if __name__ == "__main__":
    main()
