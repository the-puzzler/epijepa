# EpiWM planning results

Planning success rate (%). EpiWM starts from the released LeWM architecture, replaces SIGReg with the
epiplexity score and adds one non-affine BatchNorm on the projector output. CEM planner, LeWM defaults (300 samples,
30 iters, top-30, horizon 5). Three eval sets, all seed 42, starts drawn from the training dataset:
- **paper-50**: the released evaluation sampler (50 tasks, goal offset 25, budget 50)
- **n=200**: 200 tasks, disjoint from the 50
- **n=500**: 500 tasks, evaluated in 10 chunks of 50

EpiWM uses the same encoder objective in all environments, with λ chosen per environment.

## Success rates

Each method lists its number of seeds. Individual seed scores follow each mean in parentheses.

| Env | Budget | Method | paper-50 | n=200 | n=500 |
|---|---|---|---|---|---|
| TwoRoom | 30k | **Epi λ0.03** (3) | **100** (100/100/100) | **100** (100/100/100) | **99.9** (100/100/99.8) |
| | 30k | SIGReg retrain (2) | 89 (88/90) | 85.5 (86.5/84.5) | 87.9 (89.0/86.8) |
| | ~200k | Released | 86 | 85.0 | 82.8 |
| Push-T | 60k | **Epi λ0.1** (3) | 92 (92/94/90) | **89.3** (88.5/89.5/90.0) | 88.5 (88.4/88.0/89.0) |
| | 60k | SIGReg retrain (2) | 89 (92/86) | 87.3 (88.0/86.5) | **88.6** (88.4/88.8) |
| | ~200k | Released | 96 | 83.5 | 84.6 |
| Cube | 60k | **Epi λ0.03** (3) | 70.7 (72/66/74) | **73.2** (74.5/69.5/75.5) | **71.5** (72.2/69.8/72.4) |
| | 100k | Epi λ0.03 (1) | 78 | 75.0 | 73.0 |
| | 60k | SIGReg retrain (2) | 73 (74/72) | 64.0 (64.5/63.5) | 65.5 (66.0/65.0) |
| | ~200k | Released | 68 | 63.0 | 66.0 |
| Reacher | 60k | Epi λ0.3 (3) | 54.7 (58/50/56) | 54.7 (57.5/50.5/56.0) | 58.1 (57.6/60.4/56.4) |
| | 100k | Epi λ0.3 (3) | 64.0 (68/60/64) | 58.3 (61.0/53.5/60.5) | 63.4 (66.8/60.0/63.4) |
| | 200k | **Epi λ0.3** (3) | **72.7** (76/70/72) | **72.0** (72.0/70.0/74.0) | **73.5** (76.6/69.4/74.4) |
| | 200k | SIGReg retrain (2) | 60 (60/60) | 62.5 (62.0/63.0) | 62.2 (63.6/60.8) |
| | ~200k | Released | 52 | 62.0 | 60.8 |

SIGReg retrain = LeWM's recipe (SIGReg weight 0.09) trained by us at the same budget as Epi.

Released scores are measured with the released CEM-30 configuration. The blog separately reports the audited
CEM-10 result for released Cube on paper-50 (72%, compared with 68% here).

Reacher evaluations use the released random-policy dataset. The LeWM paper describes SAC-collected data, so
these scores are not directly comparable to its reported 86%.

[All individual scores](https://huggingface.co/basilboy/epiwm/blob/main/analysis/scores/all_scores.csv) · [Blog post](https://the-puzzler.github.io/blog/epiwm/)
