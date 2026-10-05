# Analysis: EpiWM vs the released LeWM checkpoints

All comparisons are between our best EpiWM checkpoint per environment and the official LeWM checkpoint. The
scripts are re-runnable from this folder with `$STABLEWM_HOME` pointing at LeWM's datasets and the checkpoints
(`common.py` lists exactly which).

**The bulk data is on Hugging Face,** under [basilboy/epiwm](https://huggingface.co/basilboy/epiwm/tree/main/analysis)
(`analysis/`). That covers the embedding and trajectory CSVs, the episode videos and animations, the PCA bases, the
probe JSONs, the training logs and every score. The figures and small summaries stay in this folder. To put the data
back in place:

```bash
hf download basilboy/epiwm --include "analysis/*" --local-dir ..    # from worldmodel/analysis -> worldmodel/analysis/...
```

| Path (on Hugging Face unless marked here) | What it is |
|---|---|
| `scores/scorecard.md` (here) | per-seed and mean success on all three evaluation sets |
| `scores/all_scores.csv` | every planning result we produced (see below) |
| `loss_curves.png` (here), `training_logs/*.csv` | training curves of the reported config, 3 seeds per environment |
| `embeddings/embeddings_<env>.png` (here), `embeddings/embeddings_<env>.csv` | PCA and t-SNE of 3000 random frames per environment, both models |
| `embeddings/trajectories_<env>.png` (here), `embeddings/trajectories_<env>.csv`, `embeddings/videos/`, `embeddings/animations/` | one full episode per environment moving through each model's PCA space, with its video and a side-by-side animation |
| `embeddings/pca_basis_<env>.npz`, `pca_variance_<env>.csv` | the PCA bases, to project anything else |
| `probes/probes_summary.csv` (here), `probes/probes.json` | representation probes on all four environments, both models |
| `probes/variant_search_{pusht,tworoom}.json` | the probes we ran while choosing the method (all variants, see below) |

## Scores (`scores/`)

`all_scores.csv` has one row per (run, checkpoint, eval set), all with LeWM's default CEM planner and evaluation. Its
columns are:

| Column | Meaning |
|---|---|
| `env`, `run`, `seed`, `step` | where the number comes from (`step` is the checkpoint) |
| `variant` | `epi` is the main method; rows marked `ablation: ...` are the alternatives we tried; `sigreg` is LeWM's recipe retrained by us. The reported SIGReg retrain is `tw_sigreg_0.09`, `pusht_sigreg_0.09_60k`, `cube_sigreg_0.09` and `rch_sigreg_0.09_200k`, plus their `_seed2` runs, at the same budget as EpiWM; the other `sigreg` runs are shorter schedules |
| `lam`, `max_steps` | the λ and the training budget (the cosine schedule spans `max_steps`) |
| `eval_set` | `paper50` = LeWM's own evaluation (50 episodes, seed 42); `n200` and `n500` = the same sampler with more starts |
| `success_rate`, `n_episodes` | the score in % and the number of episodes behind it |

`main == True` marks the main method, our SIGReg retrains and the released checkpoints.

## Embeddings and trajectories (`embeddings/`)

The latent is the projector output, which is what the planner compares to the goal. For each environment we draw
3000 random dataset frames, encode them with both models, and fit a 50-component PCA per model. t-SNE runs on those
50 PCs (perplexity 30, seed 0). Colours are the visible parts of the true state:

- TwoRoom: agent x/y.
- Push-T: agent x/y, block x/y, block angle.
- Cube: block position and yaw, effector position, gripper opening.
- Reacher: finger x/y and the two joint angles.

TwoRoom's and Reacher's targets are not rendered in the frame, so they are left out.

To reconstruct the trajectories against the video:

- `trajectories_<env>.csv` holds one full episode, one row per frame (TwoRoom 4748, Push-T 630, Cube 348, Reacher 348). `episode` and `step` identify the frame, and row
  `step == i` is frame `i` of `videos/<env>_ep<episode>.mp4` (10 fps).
- The true state comes next, then `epiwm_pca{1,2,3}` and `released_lewm_pca{1,2,3}`. These are coordinates in the
  same PCA bases as `embeddings_<env>.csv`, so the episodes land on the scatter plots.
- `trajectories_<env>.png` is a quick overlay of the episode on the scatter.
- `python animate_trajectory.py <env>` renders the episode video side by side with the moving point in both PCA
  spaces (the rendered ones are in `embeddings/animations/` on Hugging Face). TwoRoom episode 4748 crosses the door between the rooms.
- `pca_basis_<env>.npz` has `<model>_mean`, `<model>_components` (50×192) and `<model>_explained_variance_ratio`.
  The coordinates are `(z - mean) @ components.T`.

## Probes (`probes/`)

`probes.py` runs the same probes on both models, using 2400 random 6-frame windows per environment (2000 to fit, 400
to test):

- **linear R² / MLP R²:** how well the visible true state can be decoded from z_t, averaged over the state
  dimensions. Angles are probed as (sin, cos); for the cube this is modulo its 90° symmetry.
- **rollout err:** open-loop prediction for 5 steps with the true actions. The state is decoded from the predicted
  latents and compared with the true state, as mean |error| in state-std units.
- **goal-rank ρ:** the Spearman correlation between ‖z_t − z_goal‖ and steps-to-goal along real trajectories. This
  asks whether latent distance orders states by how far they are from the goal, which matters because CEM plans
  with this distance.
- **effective rank** of z, and the share of variance in its top-2 PCs.

| Environment | Model | linear R² | MLP R² | rollout err t=1 / t=5 | goal-rank ρ | eff. rank | top-2 PC var |
|---|---|---|---|---|---|---|---|
| tworoom | epiwm | 0.998 | 1.000 | 0.041 / 0.070 | 0.670 | 34 | 0.81 |
| tworoom | released_lewm | 0.979 | 1.000 | 0.477 / 0.501 | 0.270 | 114 | 0.05 |
| pusht | epiwm | 0.956 | 0.982 | 0.133 / 0.131 | 0.879 | 181 | 0.04 |
| pusht | released_lewm | 0.938 | 0.983 | 0.179 / 0.201 | 0.924 | 116 | 0.04 |
| cube | epiwm | 0.725 | 0.645 | 0.300 / 0.298 | 0.457 | 92 | 0.23 |
| cube | released_lewm | 0.767 | 0.648 | 0.282 / 0.291 | 0.726 | 126 | 0.03 |
| reacher | epiwm | 0.999 | 1.000 | 0.063 / 0.079 | 0.469 | 181 | 0.03 |
| reacher | released_lewm | 0.997 | 1.000 | 0.081 / 0.100 | 0.546 | 95 | 0.04 |

What the probes show:

- **State decoding** is near-perfect for both models on TwoRoom, Push-T and Reacher (MLP R² ≥ 0.98).
- **On TwoRoom,** ours is clearly better: the rollout error is about 10× lower (0.04 vs 0.48 at t=1), goal-rank ρ is
  0.67 vs 0.27, and the latent is essentially a 2-D map of the two rooms (81% of the variance in 2 PCs).
- **Rollout error** is lower for ours on Push-T and Reacher.
- **The released model wins on goal ranking** on Push-T, Cube and Reacher, and is slightly better on most Cube
  probes (linear R², rollout). So the better planning of our models is not explained by any single probe; rollout
  accuracy on TwoRoom, Push-T and Reacher is the most consistent advantage.
- **Cube's** mean R² is lower for both models because the block yaw is hard to read, even modulo its symmetry.

`variant_search_{pusht,tworoom}.json` are the probes from method selection. They cover the released model, our
SIGReg retrain and every epiplexity form we tried:

- `epi_*`: the main method at each λ.
- `epipredmlp_*`: an MLP reservoir on the predictor.
- `epiboth*`: encoder and predictor forms combined.
- `enc200k_*`: longer training.
- `enc60k_s*`: the reported Push-T config, 3 seeds.

Each entry has the same fields as `probes.json` (linear/MLP R², rollout error, goal-rank ρ, effective rank). The
encoder form at λ = 0.1 gave the best combination of decodability and rollout accuracy on Push-T, which is part of why
we chose it.

## Reproducing the analysis

```bash
python collect.py <research results dir> <released eval logs dir>   # scores/all_scores.csv, training_logs/
python loss_curves.py                                                # loss_curves.png
python embeddings.py                                                 # embeddings/ (GPU)
python probes.py                                                     # probes/ (GPU)
```

`collect.py` reads our raw research logs, which are not in the repo; its outputs are. The other scripts only need
the checkpoints and LeWM's datasets.
