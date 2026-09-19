# Epiplexity as an anti-collapse regulariser

A joint-embedding (JEPA-style) encoder trained only to make two views of an image agree collapses
to a constant. Every method needs something that pushes back. LeJEPA's SIGReg pushes the embedding
toward a prescribed distribution (an isotropic Gaussian). This repo tests a different idea: push
the embedding to be **maximally predictable from a frozen random CNN**, and specify no distribution
at all.

The score, called epiplexity here, is

    S(z) = ½ · log₂ det( I + η · WᵀW ),    W = ridge readout from reservoir features to z

on a batch, where the reservoir is a fixed randomly-initialised CNN. A collapsed embedding scores
zero. The training loss is a plain trade-off:

    loss = invariance  +  β · ( −S / S₀ )

with `S₀` calibrated once on the untrained encoder so β is on an interpretable scale. That is the
entire method: `epiplexity.py` is 60 lines.

## Results

Two chassis. In both, the **only** difference between the arms is which regulariser sits in the loss.

### CIFAR-10 — ResNet-18, 64-d embedding, 100 epochs, 3 seeds, official test split

| Regulariser | β | Backbone probe | Embedding probe | Embedding rank |
|---|---|---|---|---|
| none (collapse control) | 0 | 68.50 ± 0.35 | 48.03 | 1.0 |
| **epiplexity (ours)** | 0.3 | **76.64 ± 0.36** | 71.78 | 58.4 |
| SIGReg (LeJEPA) | 10 | 77.53 ± 0.34 | 73.17 | 24.7 |

β was chosen per method on the validation split before touching the test set. Our arm is
0.89 points behind SIGReg on three paired seeds, which is not statistically separable, and 8.1 points
above the control. The β sweep (validation) is single-peaked: 0.03 → 75.8, 0.1 → 76.7, **0.3 → 77.6**,
1 → 74.5, 3 → 56.7, 10 → 47.2. Under-weighting is cheap, over-weighting is not.

![cifar](results/cifar.png)

### Imagenette — LeJEPA's own minimal recipe, ViT-S/8 @ 128 px, 800 epochs

`imagenette/train.py` is the authors' [MINIMAL.md](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)
example verbatim (their SIGReg, encoder, projector, augmentations, optimiser, schedule, online
probe) with a `--method` switch that swaps only the regulariser. Their published number is 90.7%.

| Regulariser | λ | Online probe (their metric) | Frozen-backbone probe | Backbone rank |
|---|---|---|---|---|
| none (collapse control) | 0 | 10.96 (chance) | 23.31 | 1.5 |
| **epiplexity (ours)** | 0.1 | **86.09** (best epoch 87.16) | **87.16** | 26.6 |
| SIGReg (replication) | 0.02 | 90.42 (best epoch 90.88) | 90.22 | 38.9 |

Single seed (each run is 5–6 h and only one fits on the GPU at a time). Our regulariser needs λ about
5× theirs because its statistic is O(1) where SIGReg's is O(10). Without a regulariser the encoder
collapses in two epochs and never recovers; ours delivers 94% of SIGReg's lift above that floor.

Our λ sweep is flat: 0.05 → 85.6, 0.1 → 86.1, 0.2 → 85.5, 0.4 → 85.8, 0.8 → 85.3 (online), while the
equilibrium invariance it controls moves 79-fold. Below 0.05 training becomes unstable (λ 0.02
collapses at epoch 200 and recovers to 82.3; λ 0.01 dies). The gap to SIGReg is 2.2–3.7 points
depending on the probe and does not move with λ, with the log-det saturation η (3–300), or with the
reservoir's width, depth, pooling, or resampling — all of which were tried and all of which landed
within a point.

![imagenette](results/imagenette.png)

### Ablation: epiplexity alone

Dropping the invariance term and training the encoder to maximise the score by itself:

| Chassis | Backbone probe | Embedding probe / rank | Reference |
|---|---|---|---|
| CIFAR-10, 3 seeds | 44.21 ± 0.63 | 33.7 / 62 of 64 | collapse control 68.5, reservoir's own features 32.7 |
| Imagenette | 32.2 online, 37.4 frozen | rank 15.9 | collapse control 11.0 online, 23.3 frozen |

The encoder finds the score's global maximiser: a full-rank linear image of the random reservoir,
whose own features probe at about 33%. On CIFAR that is *below* the collapsed control, whose single
surviving dimension at least separates vehicles from animals. On Imagenette, where the projector has no
output normalisation, the encoder also inflates the output scale without bound (the log-det is not
scale-invariant). Epiplexity is an anti-collapse term, not a learning objective; the invariance term
supplies all of the content and the score only keeps it from collapsing. `results/pure_epiplexity.json`.

### What the two experiments say

Maximising predictability from a frozen random CNN prevents collapse and gets within 1–3 points of a
method that prescribes the embedding's distribution, while prescribing nothing. The score is a rank
promoter in the literal sense — the CIFAR embedding reaches effective rank 58 of 64 — and the outcome
is insensitive to how it is configured once the weight is above its collapse threshold.

## Running

```bash
pip install -r requirements.txt
pip install git+https://github.com/galilai-group/lejepa.git@c293d291ca87cd4fddee9d3fffe4e914c7272052  # SIGReg arm only

# CIFAR-10: all methods x betas x seeds, then validation selection, then official test. ~7 GPU-hours.
python cifar/train.py --out runs/cifar --stage test
python cifar/analyze.py --runs runs/cifar

# Imagenette (download imagenette2-160 from fastai; train/ and val/ class folders). ~5.5 GPU-hours per run.
python imagenette/train.py --out runs/imagenette/sigreg --data data/imagenette2-160 --method sigreg --lamb 0.02
python imagenette/train.py --out runs/imagenette/align  --data data/imagenette2-160 --method align
python imagenette/train.py --out runs/imagenette/epi    --data data/imagenette2-160 --method epi --lamb 0.1
python imagenette/probe.py  --data data/imagenette2-160 runs/imagenette/*
python imagenette/analyze.py --runs runs/imagenette
```

Each Imagenette run needs ~45 GB of GPU memory (256 images × 4 views × 256 tokens). `--grad_ckpt`
trades speed for memory. `results/` holds the JSON tables behind the figures above.

## Layout

```
epiplexity.py          the score: reservoir, ridge readout, log-det
cifar/train.py         ResNet-18 chassis, methods align | epi | sigreg, selection + test protocol
cifar/analyze.py
imagenette/train.py    LeJEPA MINIMAL.md chassis (verbatim) with the method switch
imagenette/probe.py    frozen-backbone linear + kNN probe
imagenette/analyze.py
results/               curated result tables and figures
```

## Notes

- CIFAR runs in float32 with TF32 disabled; the log-det is numerically delicate. Imagenette follows
  the authors' bfloat16 recipe for the backbone and computes the score in float64 from the projector output.
- The Imagenette online probe trains jointly with the encoder; the frozen-backbone probe refits a clean
  classifier on the finished checkpoint. They disagree about small differences between our arms
  (rank correlation 0.45) and agree on the replication. Report the frozen one.
- Reservoir: 4 conv stages for 32 px inputs, 6 for 128 px, channel-normalised, ELU, pooled to 2×2.
  Width 64 keeps the ridge readout well-posed on a 256-sample batch.
