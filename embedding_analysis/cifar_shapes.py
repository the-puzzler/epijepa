"""Shape of the learned CIFAR-10 embeddings: epiplexity vs SIGReg vs the collapse control.

Per arm: per-coordinate marginals against N(0,1), coordinate correlations, class-mean cosine
similarity, t-SNE by class (64-d embedding and 512-d backbone), covariance spectrum, norm and
pairwise-cosine distributions, plus a metrics table (kurtosis, skew, KS distance from Gaussian,
effective rank, correlation, kNN, silhouette). Figures use the first run of each arm; metrics
average over all runs given.

    python embedding_analysis/cifar_shapes.py --archive data/cifar-10-python.tar.gz \
        --align  runs/cifar/align_b0_s0  runs/cifar/align_b0_s1 \
        --epi    runs/cifar/epi_b0.3_s0  runs/cifar/epi_b0.3_s1 \
        --sigreg runs/cifar/sigreg_b10_s0 runs/cifar/sigreg_b10_s1
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cifar"))
import train as T  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
OUT = ROOT / "embedding_analysis"


def shape_stats(Z, y, Zt, yt):
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    cov = np.cov(Z.T); e = np.linalg.eigvalsh(cov).clip(0); p = e / max(e.sum(), 1e-30)
    rng = np.random.default_rng(1); W = rng.standard_normal((Z.shape[1], 64)); W /= np.linalg.norm(W, axis=0)
    pr = Zc @ W; pr = (pr - pr.mean(0)) / (pr.std(0) + 1e-12)
    sub = np.random.default_rng(0).choice(len(Z), min(2000, len(Z)), replace=False)
    mu, sd = Zt.mean(0), Zt.std(0) + 1e-12
    n = np.linalg.norm(Z - Z.mean(0), axis=1)
    return dict(
        eff_rank=float(np.exp(-(p * np.log(p + 1e-30)).sum())), top1_var_frac=float(p.max()),
        excess_kurtosis=float(np.mean(stats.kurtosis(Zc, 0))), abs_skew=float(np.mean(np.abs(stats.skew(Zc, 0)))),
        frac_coords_heavy_tailed=float((stats.kurtosis(Zc, 0) > 1).mean()),
        ks_vs_gaussian=float(np.mean([stats.kstest(pr[:, j], "norm").statistic for j in range(64)])),
        mean_abs_corr=float(np.abs(np.corrcoef(Z.T) - np.eye(Z.shape[1])).sum() / (Z.shape[1] * (Z.shape[1] - 1))),
        norm_cv=float(n.std() / n.mean()),
        knn10_acc=float(KNeighborsClassifier(10).fit((Zt - mu) / sd, yt).score((Z - mu) / sd, y)),
        silhouette=float(silhouette_score(Zc[sub], y[sub])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="data/cifar-10-python.tar.gz")
    for arm in ["align", "epi", "sigreg"]:
        p.add_argument(f"--{arm}", nargs="+", required=True, help="run directories (checkpoint.pt inside)")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data, labels, _ = T.load_data(a.archive)
    order = np.random.default_rng(746).permutation(50000)
    vx, vy = data[order[45000:]], labels[order[45000:]]
    tx, ty = data[order[:45000][:10000]], labels[order[:45000][:10000]]
    arms = [("align", "Collapse control (no regulariser)"), ("epi", "Epiplexity"), ("sigreg", "SIGReg")]
    metrics, vis = {}, {}
    for arm, title in arms:
        rows = []
        for i, d in enumerate(getattr(a, arm)):
            m = T.ResNet18(bn=arm != "sigreg").to(dev)
            m.load_state_dict(torch.load(Path(d) / "checkpoint.pt", map_location=dev, weights_only=False)["model"]); m.eval()
            h, z = T.extract(m, vx, dev); ht, zt = T.extract(m, tx, dev)
            rows.append(dict(embedding=shape_stats(z, vy, zt, ty), backbone=shape_stats(h, vy, ht, ty)))
            if i == 0: vis[title] = dict(h=h, z=z)
        metrics[arm] = dict(runs=[str(d) for d in getattr(a, arm)], per_run=rows,
                            mean={k: {s: float(np.mean([r[k][s] for r in rows])) for s in rows[0][k]} for k in ["embedding", "backbone"]})
        e = metrics[arm]["mean"]["embedding"]
        print(f"{title:36s} rank {e['eff_rank']:5.1f}  kurt {e['excess_kurtosis']:+6.2f}  KS {e['ks_vs_gaussian']:.3f}  |corr| {e['mean_abs_corr']:.3f}  knn {e['knn10_acc']*100:5.1f}", flush=True)
    (OUT / "cifar_shape_metrics.json").write_text(json.dumps(metrics, indent=1))

    titles = list(vis); cmap = plt.get_cmap("tab10")
    # --- panels: correlations, marginals, class-mean cosine
    fig, axes = plt.subplots(3, len(titles), figsize=(4.4 * len(titles), 11))
    for j, t in enumerate(titles):
        z = vis[t]["z"]; zc = (z - z.mean(0)) / (z.std(0) + 1e-12)
        C = np.corrcoef(z.T); o = np.argsort(PCA(1).fit(zc).components_[0])
        im = axes[0, j].imshow(C[o][:, o], cmap="RdBu_r", vmin=-1, vmax=1); axes[0, j].set_title(f"{t}\ncoordinate correlations", fontsize=10); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        xs = np.linspace(-4, 4, 200)
        for k in np.linspace(0, z.shape[1] - 1, 8).astype(int): axes[1, j].hist(zc[:, k], bins=60, range=(-4, 4), density=True, histtype="step", alpha=.7)
        axes[1, j].plot(xs, stats.norm.pdf(xs), "k--", lw=1.2, label="N(0,1)"); axes[1, j].set_title("8 coordinate marginals (standardised)", fontsize=10); axes[1, j].legend(fontsize=8); axes[1, j].set_ylim(0, 1.0)
        means = np.stack([zc[vy == k].mean(0) for k in range(10)]); mn = means / np.linalg.norm(means, axis=1, keepdims=True)
        im2 = axes[2, j].imshow(mn @ mn.T, cmap="viridis", vmin=-1, vmax=1); axes[2, j].set_xticks(range(10)); axes[2, j].set_xticklabels(CLASSES, rotation=90, fontsize=7); axes[2, j].set_yticks(range(10)); axes[2, j].set_yticklabels(CLASSES, fontsize=7); axes[2, j].set_title("class-mean cosine similarity", fontsize=10)
    fig.colorbar(im, ax=axes[0, :], shrink=.6, pad=.01); fig.colorbar(im2, ax=axes[2, :], shrink=.6, pad=.01)
    fig.savefig(OUT / "cifar_shape_panels.png", dpi=150, bbox_inches="tight")
    # --- t-SNE
    fig, axes = plt.subplots(2, len(titles), figsize=(4.2 * len(titles), 8.4))
    for j, t in enumerate(titles):
        for i, key in enumerate(["z", "h"]):
            X = vis[t][key]; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
            Y = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(X)
            axes[i, j].scatter(Y[:, 0], Y[:, 1], c=vy, cmap="tab10", s=3, alpha=.7, linewidths=0); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            axes[i, j].set_title(f"{t}\n{'embedding (64-d)' if key == 'z' else 'backbone (512-d)'}", fontsize=10)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=CLASSES[k]) for k in range(10)], loc="lower center", ncol=10, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(OUT / "cifar_tsne.png", dpi=150)
    # --- spectrum, norms, pairwise cosine
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for t in titles:
        z = vis[t]["z"]; e = np.linalg.eigvalsh(np.cov(z.T))[::-1]; axes[0].semilogy(np.arange(1, len(e) + 1), np.maximum(e / e.sum(), 1e-9), label=t)
        zc = z - z.mean(0); n = np.linalg.norm(zc, axis=1); axes[1].hist(n / n.mean(), bins=60, range=(0, 2.5), histtype="step", density=True, label=t)
        zn = zc / np.linalg.norm(zc, axis=1, keepdims=True); s = zn[:1500]; axes[2].hist((s @ s.T)[np.triu_indices(1500, 1)], bins=80, range=(-1, 1), histtype="step", density=True, label=t)
    axes[0].set(title="Embedding covariance spectrum", xlabel="component", ylabel="fraction of variance"); axes[1].set(title="Centered embedding norm / mean norm", xlabel="relative norm"); axes[2].set(title="Pairwise cosine similarity (centered)", xlabel="cosine")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "cifar_spectra.png", dpi=150)
    print("saved figures and cifar_shape_metrics.json to", OUT)


if __name__ == "__main__":
    main()
