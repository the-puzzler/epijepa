"""Side-by-side animation of one episode: the env video next to the same moment in each model's PCA space.
Reads only the files written by embeddings.py (videos, trajectories_<env>.csv, embeddings_<env>.csv).
usage: python animate_trajectory.py <env> [episode]   -> embeddings/animations/<env>_ep<episode>_pca.mp4
"""
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
MODELS = [("epiwm", "EpiWM (ours)"), ("released_lewm", "released LeWM")]


def main(env, episode=None):
    traj = pd.read_csv(f"{HERE}/trajectories_{env}.csv")
    cloud = pd.read_csv(f"{HERE}/embeddings_{env}.csv")
    episode = int(episode) if episode is not None else int(traj.episode.iloc[0])
    t = traj[traj.episode == episode].sort_values("step").reset_index(drop=True)
    frames = imageio.mimread(f"{HERE}/videos/{env}_ep{episode}.mp4", memtest=False)
    assert len(frames) == len(t), (len(frames), len(t))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    img = ax[0].imshow(frames[0]); ax[0].axis("off")
    dots, trails = [], []
    for a, (tag, title) in zip(ax[1:], MODELS):
        a.scatter(cloud[f"{tag}_pca1"], cloud[f"{tag}_pca2"], c="lightgrey", s=2, rasterized=True)
        a.plot(t[f"{tag}_pca1"], t[f"{tag}_pca2"], color="k", lw=0.4, alpha=0.25)   # whole path, faint
        trails.append(a.plot([], [], color="tab:red", lw=1.5)[0])
        dots.append(a.plot([], [], "o", color="tab:red", ms=8, mec="k")[0])
        a.set_title(title); a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_xticks([]); a.set_yticks([])
    suptitle = fig.suptitle("")
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(f"{HERE}/animations", exist_ok=True)
    out = f"{HERE}/animations/{env}_ep{episode}_pca.mp4"
    with imageio.get_writer(out, fps=10, macro_block_size=1) as w:
        for i in range(len(t)):
            img.set_data(frames[i])
            for (tag, _), d, tr in zip(MODELS, dots, trails):
                d.set_data([t.loc[i, f"{tag}_pca1"]], [t.loc[i, f"{tag}_pca2"]])
                tr.set_data(t.loc[:i, f"{tag}_pca1"], t.loc[:i, f"{tag}_pca2"])
            suptitle.set_text(f"{env}, episode {episode}, step {i}/{len(t) - 1}")
            fig.canvas.draw()
            px = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            w.append_data(px[: px.shape[0] // 2 * 2, : px.shape[1] // 2 * 2])  # libx264 needs even dimensions
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
