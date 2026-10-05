"""Training curves of the reported EpiWM config per environment, 3 seeds each (training_logs/*.csv, from collect.py).
Rows: prediction loss (train), prediction loss (validation, every 10k/20k steps), epiplexity ratio S/S0 (the term
being maximised), effective rank of the embeddings. Each environment runs to its own best budget (30k/60k/60k/200k).
usage: python loss_curves.py   -> loss_curves.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"tworoom (λ=0.03, 30k)": "tw_epi_0.03", "pusht (λ=0.1, 60k)": "pusht_epi_0.1_60k",
        "cube (λ=0.03, 60k)": "cube_epi_0.03", "reacher (λ=0.3, 200k)": "rch_epi_0.3_200k"}
ROWS = [("pred", "prediction loss (train)", True), ("val_pred", "prediction loss (val)", True),
        ("epi_ratio", "epiplexity S / S0", False), ("erank", "effective rank of z", False)]

fig, ax = plt.subplots(len(ROWS), len(RUNS), figsize=(4.2 * len(RUNS), 2.8 * len(ROWS)), squeeze=False)
for j, (title, base) in enumerate(RUNS.items()):
    for f in sorted(glob.glob(f"{HERE}/training_logs/{base}*.csv")):
        run = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        seed = run.split("_seed")[1] if "_seed" in run else "1"
        for i, (col, ylab, logy) in enumerate(ROWS):
            s = d[["step", col]].dropna()
            ax[i, j].plot(s.step, s[col], lw=1, alpha=0.8, marker="o" if col == "val_pred" else None, ms=3,
                          label=f"seed {seed}")
            if logy:
                ax[i, j].set_yscale("log")
            if j == 0:
                ax[i, j].set_ylabel(ylab, fontsize=9)
    ax[0, j].set_title(title)
    ax[-1, j].set_xlabel("step")
    ax[0, j].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{HERE}/loss_curves.png", dpi=90)
print("saved loss_curves.png")
