"""Gather every planning score and training log from the research results directory into the analysis folder.

  scores/all_scores.csv    one row per (run, checkpoint, eval set) with the default LeWM CEM planner:
                           paper50 (LeWM's own eval: 50 episodes, seed 42), n200, n500 (same sampler, more starts)
  training_logs/<run>.csv  train + validation curves of the reported EpiJEPA config per environment (3 seeds)
Variant runs (alternative epiplexity forms, ablations) are included in all_scores.csv so the search is visible;
only the main method (variant == "epi") and the SIGReg control are part of the released code.
usage: python collect.py <results_dir> <released_eval_dir>
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = {"tw": "tworoom", "pusht": "pusht", "cube": "cube", "rch": "reacher"}
VARIANT = {  # run-name tag -> what it is
    "epi": "epi: frozen CNN reservoir of o_t -> z_t (main method)",
    "sigreg": "sigreg: LeWM's SIGReg (our control)",
    "epifa": "ablation: full-anchor epi, frozen joint reservoir r(o_t,a_t) -> z_t+1",
    "epiapred": "ablation: action-pred epi, frozen MLP R(z_t,a_t) -> encoder z_t+1",
    "epipredmlp": "ablation: predictor epi, frozen MLP reservoir of z_t -> predicted z_t+1",
    "epiboth": "ablation: mean of encoder and predictor epi",
    "epibothsum": "ablation: sum of encoder and predictor epi",
    "episig": "ablation: epi + SIGReg",
}
SUFFIX = {"_sgtgt": "ablation: stop-gradient on the prediction target", "_predwhiten": "ablation: whitened prediction loss",
          "_whitenout": "ablation: batch-whitened projector output", "_full": "LeWM config budget (100 epochs)"}
TRAINING_LOGS = ["tw_epi_0.03", "tw_epi_0.03_seed2", "tw_epi_0.03_seed3",
                 "pusht_epi_0.1_60k", "pusht_epi_0.1_60k_seed2", "pusht_epi_0.1_60k_seed3",
                 "cube_epi_0.03", "cube_epi_0.03_seed2", "cube_epi_0.03_seed3",
                 "rch_epi_0.3_200k", "rch_epi_0.3_200k_seed2", "rch_epi_0.3_200k_seed3"]  # the reported config, 3 seeds each


def success(path):
    """(success rate %, n episodes) of a LeWM eval log; for chunked evals pool the chunk logs."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        m = re.search(r"'success_rate': ([0-9.]+)", open(path).read())
        if m:
            s = open(path).read()
            n = len(re.search(r"'episode_successes': array\(\[(.*?)\]", s, re.S).group(1).split(","))
            return float(m.group(1)), n
    chunks = sorted(f for f in os.listdir(os.path.dirname(path) or ".") if f.startswith(os.path.basename(path)[:-4] + "_chunk"))
    succ = []
    for c in chunks:
        s = open(os.path.join(os.path.dirname(path), c)).read()
        m = re.search(r"'episode_successes': array\(\[(.*?)\]", s, re.S)
        if not m:
            return None
        succ += [x.strip() == "True" for x in m.group(1).split(",")]
    return (100 * float(np.mean(succ)), len(succ)) if succ else None


def describe(run, results):
    cfg = yaml.safe_load(open(os.path.join(results, run, "config.yaml")))
    tag = re.match(r"^[a-z]+_([a-z]+)_", run).group(1)
    variant = next((v for k, v in SUFFIX.items() if k in run), VARIANT.get(tag, tag))
    seed_m = re.search(r"_seed(\d+)", run)
    return dict(run=run, env=ENV[run.split("_")[0]], variant=variant, main=(tag in ("epi", "sigreg") and variant == VARIANT.get(tag)),
                regulariser=cfg["loss"]["reg"], lam=cfg["loss"]["epi"]["weight"] if cfg["loss"]["reg"] != "sigreg" else None,
                sigreg_weight=cfg["loss"]["sigreg"]["weight"] if cfg["loss"]["reg"] != "epi" else None,
                max_steps=cfg["max_steps"], seed=int(seed_m.group(1)) if seed_m else 1)


def main(results, released_dir):
    rows = []
    runs = sorted(r for r in os.listdir(results) if re.match(r"^(tw|pusht|cube|rch)_", r)
                  and "MISLABELED" not in r and os.path.exists(os.path.join(results, r, "config.yaml")))
    for run in runs:
        meta = describe(run, results)
        env, d = meta["env"], os.path.join(results, run)
        for f in sorted(os.listdir(d)):
            m = re.match(rf"^(eval|eval200)_{env}_step(\d+)\.log$", f)
            if m:
                r = success(os.path.join(d, f))
                if r:
                    rows.append(dict(meta, step=int(m.group(2)), eval_set="paper50" if m.group(1) == "eval" else "n200",
                                     success_rate=r[0], n_episodes=r[1]))
    for line in open(os.path.join(results, "eval500", "summary.txt")):  # n500: "env tag n=500 success_rate=X ..."
        env, tag = line.split()[:2]
        sr = float(re.search(r"success_rate=([0-9.]+)", line).group(1))
        if tag.startswith("official"):
            rows.append(dict(run=f"official_{env}", env=env, variant="released LeWM checkpoint", main=True, regulariser="sigreg",
                             step=None, eval_set="n500", success_rate=sr, n_episodes=500))
        else:
            run, step = tag.split("@")
            rows.append(dict(describe(run, results), step=int(step), eval_set="n500", success_rate=sr, n_episodes=500))
    released = {"tworoom": ("eval_official_tworoom.log", "eval200_official_tworooms.log"),
                "pusht": ("eval_official_pusht.log", "eval200_official_pusht.log"),
                "cube": ("eval50_official_cube_v5.log", "eval200_official_cube_v5.log"),
                "reacher": ("eval50_official_reacher_v5.log", "eval200_official_reacher_v5.log")}
    for env, files in released.items():
        for f, es in zip(files, ("paper50", "n200")):
            r = success(os.path.join(released_dir, f))
            if r:
                rows.append(dict(run=f"official_{env}", env=env, variant="released LeWM checkpoint", main=True,
                                 regulariser="sigreg", step=None, eval_set=es, success_rate=r[0], n_episodes=r[1]))
    df = pd.DataFrame(rows).drop_duplicates(subset=["run", "step", "eval_set"])
    os.makedirs(f"{HERE}/scores", exist_ok=True)
    df.to_csv(f"{HERE}/scores/all_scores.csv", index=False)
    print(f"{len(df)} score rows from {df.run.nunique()} runs -> scores/all_scores.csv")

    os.makedirs(f"{HERE}/training_logs", exist_ok=True)
    for run in TRAINING_LOGS:
        log = json.load(open(os.path.join(results, run, "train_log.json")))
        tr = pd.DataFrame([r for r in log["log"] if "loss" in r]).rename(columns={"gn": "grad_norm"})
        va = pd.DataFrame([r for r in log["log"] if "val_pred" in r])
        tr.merge(va, on="step", how="outer").sort_values("step").to_csv(f"{HERE}/training_logs/{run}.csv", index=False)
    print(f"{len(TRAINING_LOGS)} training logs -> training_logs/")


if __name__ == "__main__":
    main(*sys.argv[1:3])
