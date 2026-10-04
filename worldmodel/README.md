# EpiJEPA as the anti-collapse term of LeWorldModel

[LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM) learns a latent world model end-to-end from pixels and plans
in its latent space with CEM. To stop the embedding collapsing it adds **SIGReg**, which pushes the embeddings
towards an isotropic Gaussian. Here SIGReg is replaced by the **epiplexity score** from EpiJEPA (`../epiplexity.py`),
which rewards embeddings that are linearly predictable from a frozen random CNN reservoir of the same frames.
Everything else is LeWM's: the architecture of the released checkpoints, the data, the optimiser and schedule, and
the planner and evaluation.

```
SIGReg (LeWM):  loss = ||pred(z_t, a_t) - z_t+1||^2 + 0.09 * SIGReg(z)
EpiJEPA:        loss = ||pred(z_t, a_t) - z_t+1||^2 - lambda * S(z) / S0
                S  = epiplexity of the embeddings w.r.t. a frozen random CNN reservoir of the frames
                S0 = S of the untrained model (average over 4 batches)
```

The EpiJEPA model has one architectural change: a non-affine BatchNorm on the projector output
(`module.ProjectorBN`). The score grows with the embedding scale, so the scale is pinned, as in the CIFAR and
Imagenette EpiJEPA encoders. SIGReg does not need this because its N(0, I) target already fixes the scale.

## Results

Planning success rate (%) with LeWM's CEM planner. There are three evaluation sets (see *Evaluation* below), and we
report the mean over 3 seeds. "Released" is the official LeWM checkpoint (about 200k steps).

| Environment | Model (steps, seeds) | paper-50 | n=200 | n=500 |
|---|---|---|---|---|
| TwoRoom | **EpiJEPA** λ=0.03 (30k, 3) | **100** | **100** | **99.9** |
| | released LeWM | 86 | 85.0 | 82.8 |
| Push-T | **EpiJEPA** λ=0.1 (60k, 3) | 92 | **89.3** | 88.5 |
| | released LeWM | 96 | 83.5 | 84.6 |
| Cube | **EpiJEPA** λ=0.03 (60k, 3) | 70.7 | **73.2** | **71.5** |
| | released LeWM | 68 | 63.0 | 66.0 |
| Reacher | **EpiJEPA** λ=0.3 (200k, 3) | **72.7** | **72.0** | **73.5** |
| | released LeWM | 52 | 62.0 | 60.8 |

On n=500, with Fisher's exact test over pooled seeds:

- **Against the released checkpoints,** EpiJEPA is better on all four environments:
  - TwoRoom: 99.9 vs 82.8.
  - Cube: 71.5 vs 66.0 (p = 0.02).
  - Push-T: 88.5 vs 84.6.
  - Reacher: 73.5 vs 60.8 at the same ~200k-step budget (p ≈ 1e-7).

The paper-50 set is noisy. For example, the released Push-T checkpoint scores 96 on it but 83.5 and 84.6 on n=200
and n=500. Per-seed numbers, all checkpoints and the λ sweeps are in [`analysis/`](analysis/).

The best checkpoint per environment is on Hugging Face: [basilboy/epijepa-lewm](https://huggingface.co/basilboy/epijepa-lewm).

### What the latent looks like

![Cube episode: video next to the moving point in each model's PCA space](assets/cube_pca.gif)

This is one Cube episode, shown next to the same moment in each model's planning latent (the first two principal
components, over 3000 random frames). In EpiJEPA's latent the arm and block move along a smooth, structured manifold.
The released LeWM latent is a more diffuse cloud. The episode data behind this (frame-by-frame coordinates, true
state and video), the same for every environment, plus all embeddings, probes, training logs and scores, is in the
Hugging Face repo under [`analysis/`](https://huggingface.co/basilboy/epijepa-lewm/tree/main/analysis). See
[`analysis/`](analysis/) for the figures and how to use it.

## Reproducing

### Setup

The code was run with torch 2.13.0 (2.14's cuDNN was broken on our machine), torchvision 0.28, transformers 5.17,
stable-worldmodel 0.1.1, stable-pretraining 0.1.7, hydra-core 1.3.7, h5py, hdf5plugin, scikit-learn and mujoco 3.14.
See `requirements.txt`.

Download LeWM's datasets from Hugging Face (`quentinll/lewm-{tworooms,pusht,cube,reacher}`) into
`$STABLEWM_HOME/datasets`. Then run everything from this folder:

```bash
export STABLEWM_HOME=/path/to/stablewm MUJOCO_GL=egl
```

### Training

The steps below are the best training budget per environment. They matter for reproduction, because the cosine
schedule spans `max_steps`. Planning success is not monotone in training length: on Push-T both regularisers peak
around 60k steps (EpiJEPA at 200k reaches only 81.5), while Reacher keeps improving up to 200k.

| Environment | `data=` | λ | Steps | Our wall clock (1 GPU, shared 4 ways) |
|---|---|---|---|---|
| TwoRoom | `tworoom` | 0.03 | 30,000 | ~5 h |
| Push-T | `pusht` | 0.1 | 60,000 | ~11 h |
| Cube | `ogb` | 0.03 | 60,000 | ~12 h |
| Reacher | `dmc` | 0.3 | 200,000 | ~30–40 h |

```bash
python train.py data=tworoom model=epi loss.reg=epi loss.epi.weight=0.03 max_steps=30000  run_name=tw_epi
python train.py data=pusht   model=epi loss.reg=epi loss.epi.weight=0.1  max_steps=60000  run_name=pusht_epi
python train.py data=ogb     model=epi loss.reg=epi loss.epi.weight=0.03 max_steps=60000  run_name=cube_epi
python train.py data=dmc     model=epi loss.reg=epi loss.epi.weight=0.3  max_steps=200000 run_name=reacher_epi
# our replication of LeWM (SIGReg, weight 0.09): same command with model=sigreg loss.reg=sigreg
python train.py data=pusht   model=sigreg loss.reg=sigreg max_steps=60000 run_name=pusht_sigreg
```

Seeds 2 and 3 use `seed=2` and `seed=3`; the default seed is LeWM's 3072. We ran with `num_workers=4`. Checkpoints go
to `$STABLEWM_HOME/checkpoints/<run_name>/weights_step<N>.pt`, and logs go to `runs/<run_name>/train_log.json`.

### Evaluation

```bash
python eval.py --config-name=tworoom.yaml policy=tw_epi/weights_step30000.pt                       # paper-50
python eval.py --config-name=tworoom.yaml policy=tw_epi/weights_step30000.pt eval.num_eval=200     # n=200
python eval.py --config-name=cube.yaml policy=cube_epi/weights_step60000.pt eval.num_eval=200 +eval_chunk=0/4  # chunks 0..3
```

Here `eval.num_eval=500` with `+eval_chunk=i/10` gives n=500. Pooling the chunks reproduces the unchunked start set
exactly.

The released LeWM checkpoints (`quentinll/lewm-*`) were saved with transformers 4. Convert them once with
`python convert_hf4_ckpt.py weights.pt weights_v5.pt`, and put the result in a checkpoint folder with its
`config.json`.

### Check that this code reproduces the results

`train.py` is a cleaned-up version of the research script that produced the numbers above, with the ablations
removed. We checked it by retraining TwoRoom from scratch with it.

- It gives the same S0 (256.832) and the same step-1 losses as the original run.
- At 10k steps it tracks the original (prediction loss 0.0042 vs 0.0041; effective rank 19.3 vs 19.7).
- After 30k steps it plans at 100% on paper-50 and 99.5% on n=200, against 100 / 100 for the original seeds.
- The runs differ after the first step only through random-number order.

## What comes from LeWM, and what is ours

LeWM is MIT-licensed (`LICENSE_LEWM`). These files are copied from it:

- `utils.py`: verbatim (dataset transforms and normalisers).
- `module.py`: verbatim LeWM modules, including SIGReg. Only `ProjectorBN` at the bottom is ours.
- `eval.py`: LeWM's evaluation script. Our only change is the optional `+eval_chunk` (three lines marked
  `[EpiJEPA addition]`), which splits one evaluation across several processes to save host RAM; Cube needs this at
  n≥200.
- `config/eval/*`: verbatim, covering the planner (CEM: 300 samples, 30 iterations, top-30, horizon 5, action
  block 5) and the evaluation settings.
- `config/train/data/*`: verbatim.
- `config/train/lewm.yaml`: LeWM's training config (seed, batch size, optimiser, history 3, SIGReg weight 0.09). We
  only added the `loss.reg` and `loss.epi` block.
- `config/train/model/sigreg.yaml`: the exact architecture of the released checkpoints, read from their
  `config.json`, using stable-worldmodel's classes. `epi.yaml` is identical except for the projector
  (`module.ProjectorBN`).
- **Evaluation sampling:** all our numbers use LeWM's own start/goal sampling in `eval.py`. It uses seed 42, takes
  start states from the dataset, sets the goal 25 steps ahead in the same episode, and gives a budget of 50 steps.
  paper-50 is LeWM's default `num_eval=50`. n=200 and n=500 change only `num_eval` (the n=200 starts differ from the
  50, though a few episodes overlap). Note that with this protocol the start states come from the training data.

Ours:

- `train.py`: LeWM trains with Lightning (`train.py` in their repo). This is the same loop in plain PyTorch, with the
  same model, data, AdamW (lr 5e-5, wd 1e-3), 1% warmup and cosine schedule, grad-clip 1.0, bf16 and batch 128. It
  adds the choice of regulariser.
- The epiplexity regulariser (`../epiplexity.py`, unchanged from the EpiJEPA image experiments) and `ProjectorBN`.
- `analysis/`: probes, embeddings, score collection and plots.

TwoRoom note: the LeWM paper's text gives a history length of 1 for TwoRoom, but the released checkpoint uses 3. We
use 3 everywhere, matching the release.
