"""Shape of the learned Imagenette embeddings: epiplexity vs SIGReg vs the collapse control.

Projector output (16-d, what both regularisers act on): all coordinate marginals against N(0,1),
coordinate correlations, a random 2-D projection. Backbone (512-d): t-SNE by class and covariance
spectrum. Metrics per arm to imagenette_shape_metrics.json.

    python embedding_analysis/imagenette_shapes.py --data data/imagenette2-160 \
        --align runs/imagenette/align --epi runs/imagenette/epi --sigreg runs/imagenette/sigreg
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "imagenette"))
from train import ViTEncoder, Views  # noqa: E402

CLASSES = ["tench", "springer", "cassette", "chainsaw", "church", "horn", "truck", "pump", "golf", "chute"]
OUT = ROOT / "embedding_analysis"


def erank(X):
    e = np.linalg.eigvalsh(np.cov(X.T)).clip(0); p = e / max(e.sum(), 1e-30)
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--proj_dim", type=int, default=16)
    for arm in ["align", "epi", "sigreg"]: p.add_argument(f"--{arm}", required=True, help="run directory (checkpoint.pt inside)")
    a = p.parse_args()
    train = DataLoader(Views(a.data, "train", V=1), batch_size=128, num_workers=4, shuffle=True)
    val = DataLoader(Views(a.data, "val", V=1), batch_size=128, num_workers=4)
    arms = [("align", "Collapse control"), ("epi", "Epiplexity"), ("sigreg", "SIGReg")]
    data, metrics = {}, {}
    for arm, title in arms:
        net = ViTEncoder(proj_dim=a.proj_dim).cuda()
        net.load_state_dict(torch.load(Path(getattr(a, arm)) / "checkpoint.pt", map_location="cuda", weights_only=False)["net"]); net.eval()
        PJ, EB, Y = [], [], []
        with torch.no_grad():
            for i, (vs, _) in enumerate(train):
                if i >= 24: break
                with torch.autocast("cuda", dtype=torch.bfloat16): _, pr = net(vs.cuda())
                PJ.append(pr[0].float().cpu().numpy())
            for vs, y in val:
                with torch.autocast("cuda", dtype=torch.bfloat16): e = net.backbone(vs.cuda().flatten(0, 1))
                EB.append(e.float().cpu().numpy()); Y.append(y.numpy())
        proj, emb, y = np.concatenate(PJ), np.concatenate(EB), np.concatenate(Y)
        data[title] = dict(proj=proj, emb=emb, y=y)
        zc = (proj - proj.mean(0)) / (proj.std(0) + 1e-9); D = proj.shape[1]
        metrics[arm] = dict(run=getattr(a, arm),
            projector=dict(eff_rank=erank(proj), excess_kurtosis=float(np.mean(stats.kurtosis(zc, 0))), abs_skew=float(np.mean(np.abs(stats.skew(zc, 0)))),
                           ks_vs_gaussian=float(np.mean([stats.kstest(zc[:, j], "norm").statistic for j in range(D)])),
                           mean_abs_corr=float(np.abs(np.corrcoef(proj.T) - np.eye(D)).sum() / (D * (D - 1)))),
            backbone=dict(eff_rank=erank(emb), excess_kurtosis=float(np.mean(stats.kurtosis((emb - emb.mean(0)) / (emb.std(0) + 1e-9), 0)))))
        m = metrics[arm]
        print(f"{title:18s} proj: rank {m['projector']['eff_rank']:5.2f}/{D} kurt {m['projector']['excess_kurtosis']:+6.2f} KS {m['projector']['ks_vs_gaussian']:.3f} |corr| {m['projector']['mean_abs_corr']:.3f} | backbone rank {m['backbone']['eff_rank']:5.1f}", flush=True)
        del net; torch.cuda.empty_cache()
    (OUT / "imagenette_shape_metrics.json").write_text(json.dumps(metrics, indent=1))

    titles = list(data); cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for j, t in enumerate(titles):
        z = data[t]["proj"]; zc = (z - z.mean(0)) / (z.std(0) + 1e-9); xs = np.linspace(-4, 4, 200)
        for k in range(z.shape[1]): axes[0, j].hist(zc[:, k], bins=60, range=(-4, 4), density=True, histtype="step", alpha=.6, lw=.8)
        axes[0, j].plot(xs, stats.norm.pdf(xs), "k--", lw=1.4, label="N(0,1)"); axes[0, j].set_ylim(0, 0.9); axes[0, j].legend(fontsize=8)
        axes[0, j].set_title(f"{t}\nall {z.shape[1]} projector coordinates, standardised", fontsize=10)
        C = np.corrcoef(z.T); o = np.argsort(PCA(1).fit(zc).components_[0])
        im = axes[1, j].imshow(C[o][:, o], cmap="RdBu_r", vmin=-1, vmax=1); axes[1, j].set_title("coordinate correlations", fontsize=10); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        W = np.random.default_rng(0).standard_normal((z.shape[1], 2)); W /= np.linalg.norm(W, axis=0); pr = zc @ W
        axes[2, j].scatter(pr[:, 0], pr[:, 1], s=2, alpha=.35); axes[2, j].set_title("random 2-D projection of the projector output", fontsize=10); axes[2, j].set_xlim(-4, 4); axes[2, j].set_ylim(-4, 4); axes[2, j].set_aspect("equal")
    fig.colorbar(im, ax=axes[1, :], shrink=.7, pad=.01); fig.savefig(OUT / "imagenette_projector_shape.png", dpi=140, bbox_inches="tight")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for j, t in enumerate(titles):
        X = data[t]["emb"]; X = (X - X.mean(0)) / (X.std(0) + 1e-9); y = data[t]["y"]
        Yt = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(X)
        axes[0, j].scatter(Yt[:, 0], Yt[:, 1], c=y, cmap="tab10", s=3, alpha=.7, linewidths=0); axes[0, j].set_xticks([]); axes[0, j].set_yticks([]); axes[0, j].set_title(f"{t}\nbackbone (512-d), t-SNE by class", fontsize=10)
        e = np.linalg.eigvalsh(np.cov(X.T))[::-1].clip(1e-12); axes[1, j].semilogy(np.arange(1, 65), e[:64] / e.sum()); axes[1, j].set_title("backbone covariance spectrum, top 64", fontsize=10); axes[1, j].set_xlabel("component"); axes[1, j].set_ylabel("fraction of variance"); axes[1, j].grid(alpha=.2); axes[1, j].set_ylim(1e-4, 1)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=CLASSES[k]) for k in range(10)], loc="lower center", ncol=10, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(OUT / "imagenette_backbone.png", dpi=140)
    print("saved figures and imagenette_shape_metrics.json to", OUT)


if __name__ == "__main__":
    main()
