"""PCA and t-SNE of the planning latent (projector output) for our best EpiWM model and the released LeWM model,
plus whole episodes projected into each model's PCA space (with the episode videos).

For each environment, N random dataset frames are encoded by both models and a PCA is fitted per model. We save
  embeddings_<env>.csv      one row per random frame: dataset row, episode, step, true state labels, and per model
                            pca1/pca2/tsne1/tsne2 (t-SNE runs on the top-50 PCs)
  embeddings_<env>.png      rows = state labels (colour), columns = model x {PCA, t-SNE}
  pca_variance_<env>.csv    explained-variance ratio of the first 30 PCs per model
  pca_basis_<env>.npz       per model: mean (192,), components (50, 192), explained_variance_ratio (50,)
  trajectories_<env>.csv    the showcased episode (EPISODES), one row per frame: episode, step (= frame index in the video), true
                            state, and per model pca1/pca2/pca3 in that model's PCA basis above
  videos/<env>_ep<episode>.mp4   the frames of that episode, 10 fps, frame i = step i
  trajectories_<env>.png    the episode drawn on top of the PCA scatter (colour = time)
usage: python embeddings.py [env ...]     (default: all four; run with $STABLEWM_HOME set)
"""
import os
import sys

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_worldmodel as swm
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # worldmodel/ (module.ProjectorBN)
from common import DATA, ENVS, encode, load_rows, sample_rows  # noqa: E402

N, FPS = 3000, 10
EPISODES = {"tworoom": [4748], "pusht": [630], "cube": [348], "reacher": [348]}  # TwoRoom 4748 crosses the door
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
MODELS = [("epiwm", "ours"), ("released_lewm", "released")]


def episode_rows(env):
    """Rows of the showcased episode(s) of this environment, as a list of (episode id, row array)."""
    with h5py.File(os.path.join(DATA, ENVS[env]["h5"]), "r") as f:
        offs, lens = f["ep_offset"][:], f["ep_len"][:]
    return [(e, np.arange(offs[e], offs[e] + lens[e])) for e in EPISODES[env]]


def run(env):
    rows = sample_rows(env, N, seed=0)
    pix, labels = load_rows(env, rows)
    episodes = [(e, *load_rows(env, r)) for e, r in episode_rows(env)]
    df = pd.DataFrame({"row": rows, **labels})
    traj = pd.concat([pd.DataFrame(lab) for _, _, lab in episodes], ignore_index=True)
    var, basis = {}, {}
    for tag, key in MODELS:
        model = swm.wm.utils.load_pretrained(ENVS[env][key]).cuda().eval()
        Z = encode(model, pix).numpy()
        Ze = np.concatenate([encode(model, p).numpy() for _, p, _ in episodes])
        del model; torch.cuda.empty_cache()
        pca = PCA(n_components=50, random_state=0).fit(Z)  # PCA centres on the random-frame mean
        P, Pe = pca.transform(Z), pca.transform(Ze)
        tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(P)
        df[f"{tag}_pca1"], df[f"{tag}_pca2"] = P[:, 0], P[:, 1]
        df[f"{tag}_tsne1"], df[f"{tag}_tsne2"] = tsne[:, 0], tsne[:, 1]
        for k in range(3):
            traj[f"{tag}_pca{k + 1}"] = Pe[:, k]
        var[tag] = pca.explained_variance_ratio_[:30]
        basis.update({f"{tag}_mean": pca.mean_, f"{tag}_components": pca.components_,
                      f"{tag}_explained_variance_ratio": pca.explained_variance_ratio_})
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    df.to_csv(f"{OUT}/embeddings_{env}.csv", index=False)
    traj.to_csv(f"{OUT}/trajectories_{env}.csv", index=False)
    pd.DataFrame(var, index=pd.RangeIndex(1, 31, name="pc")).to_csv(f"{OUT}/pca_variance_{env}.csv")
    np.savez(f"{OUT}/pca_basis_{env}.npz", **basis)
    for e, p, _ in episodes:
        imageio.mimwrite(f"{OUT}/videos/{env}_ep{e}.mp4", p, fps=FPS, macro_block_size=1)

    names = list(ENVS[env]["labels"])
    cols = [(tag, m) for tag, _ in MODELS for m in ("pca", "tsne")]
    fig, ax = plt.subplots(len(names), len(cols), figsize=(3.2 * len(cols), 3.0 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, (tag, m) in enumerate(cols):
            a = ax[i, j]
            a.scatter(df[f"{tag}_{m}1"], df[f"{tag}_{m}2"], c=df[n], s=1.5, cmap="viridis", rasterized=True)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                extra = f" (top-2 PCs {var[tag][:2].sum():.0%} var)" if m == "pca" else ""
                a.set_title(f"{tag}: {m.upper()}{extra}", fontsize=9)
            if j == 0:
                a.set_ylabel(f"colour = {n}", fontsize=9)
    fig.suptitle(f"{env}: planning latent, {N} random frames", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(f"{OUT}/embeddings_{env}.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4.5), squeeze=False)
    for j, (tag, _) in enumerate(MODELS):
        a = ax[0, j]
        a.scatter(df[f"{tag}_pca1"], df[f"{tag}_pca2"], c="lightgrey", s=1, rasterized=True)
        for e in traj.episode.unique():
            t = traj[traj.episode == e]
            a.plot(t[f"{tag}_pca1"], t[f"{tag}_pca2"], lw=0.6, color="k", alpha=0.4)
            a.scatter(t[f"{tag}_pca1"], t[f"{tag}_pca2"], c=t.step, cmap="plasma", s=4)
        a.set_title(f"{tag}: episode in PCA space (colour = step)", fontsize=9)
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{OUT}/trajectories_{env}.png", dpi=90); plt.close(fig)
    print(f"{env}: saved", flush=True)


if __name__ == "__main__":
    for e in sys.argv[1:] or list(ENVS):
        run(e)
