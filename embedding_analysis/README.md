# Embedding analysis

What the learned embeddings look like under each regulariser. Figures use seed 0; the CIFAR metrics
average three seeds. Both chassis, same three arms: the collapse control (invariance only),
epiplexity (β 0.3 on CIFAR, λ 0.1 on Imagenette), and SIGReg (β 10, λ 0.02).

## Summary

**Epiplexity embeddings are not Gaussian. They are a decorrelated, heavy-tailed cloud.**

| On the regularised embedding | Collapse control | Epiplexity | SIGReg |
|---|---|---|---|
| Effective rank, CIFAR (of 64) | 1.0 | 58.4 | 24.7 |
| Effective rank, Imagenette projector (of 16) | 2.0 | 15.2 | 15.9 |
| Excess kurtosis, CIFAR / Imagenette | -1.0 / +12.1 | +4.2 / +3.4 | +0.5 / +0.1 |
| KS distance of 1-D projections from N(0,1), CIFAR / Imagenette | 0.23 / 0.21 | 0.076 / 0.087 | 0.035 / 0.034 |
| Mean absolute coordinate correlation, CIFAR / Imagenette | 1.00 / 0.62 | 0.048 / 0.069 | 0.136 / 0.024 |
| Backbone effective rank, Imagenette | 1.1 | 22.4 | 31.1 |

- **Marginals are peaked with fat tails.** Every epiplexity coordinate rises above the Gaussian at zero
  and again beyond ±2; on CIFAR 98% of coordinates are heavy-tailed. SIGReg's sit on the Gaussian, as
  its objective demands.
- **Coordinates are nearly independent.** On CIFAR the epiplexity correlation matrix is closer to the
  identity than SIGReg's. Heavy-tailed, independent coordinates is the signature of a sparse or ICA-like
  code, and the random 2-D projection of the Imagenette projector output shows it directly: a dense
  core with radial arms, where SIGReg gives a round blob and the collapse control a curve.
- **Class structure is similar in kind.** Both give ten clean clusters in the Imagenette backbone
  t-SNE; on CIFAR both separate vehicles cleanly and mix the animals. Epiplexity's backbone spectrum
  on Imagenette decays sooner (to 1e-4 by component ~45 vs ~63), matching its lower backbone rank.

Descriptively: the score pushes toward an embedding whose coordinates are independent and sparse-looking
rather than jointly Gaussian, full-rank on the embedding and somewhat lower-rank on the backbone than
SIGReg. It prescribes no distribution and ends up with a recognisable one anyway.

## Figures

| File | Content |
|---|---|
| `cifar_shape_panels.png` | coordinate correlations, 8 marginals vs N(0,1), class-mean cosine similarity |
| `cifar_tsne.png` | t-SNE by class of the 64-d embedding and the 512-d backbone |
| `cifar_spectra.png` | covariance spectrum, norm distribution, pairwise cosine distribution |
| `imagenette_projector_shape.png` | all 16 projector marginals vs N(0,1), correlations, a random 2-D projection |
| `imagenette_backbone.png` | backbone t-SNE by class and covariance spectrum |

Metrics: `cifar_shape_metrics.json` (per run and mean, embedding and backbone), `imagenette_shape_metrics.json`.

## Regenerate

```bash
python embedding_analysis/cifar_shapes.py --archive data/cifar-10-python.tar.gz \
    --align  runs/cifar/align_b0_s0  runs/cifar/align_b0_s1  runs/cifar/align_b0_s2 \
    --epi    runs/cifar/epi_b0.3_s0  runs/cifar/epi_b0.3_s1  runs/cifar/epi_b0.3_s2 \
    --sigreg runs/cifar/sigreg_b10_s0 runs/cifar/sigreg_b10_s1 runs/cifar/sigreg_b10_s2

python embedding_analysis/imagenette_shapes.py --data data/imagenette2-160 \
    --align runs/imagenette/align --epi runs/imagenette/epi --sigreg runs/imagenette/sigreg
```
