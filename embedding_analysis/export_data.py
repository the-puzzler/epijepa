"""Export the embeddings behind the shape figures, with class labels, so they can be replotted elsewhere.

Writes embedding_analysis/data/{cifar,imagenette}_embeddings.npz (raw embeddings as float16, t-SNE
coordinates as float32, PCA scores on the first 10 components plus the explained-variance ratio,
integer labels, class names) and CSVs of the t-SNE and PCA points.
t-SNE uses the same settings as the figure scripts (standardised features, PCA init, perplexity 30, seed 0).

    python embedding_analysis/export_data.py --archive data/cifar-10-python.tar.gz --data data/imagenette2-160 \
        --cifar_align runs/cifar/align_b0_s0 --cifar_epi runs/cifar/epi_b0.3_s0 --cifar_sigreg runs/cifar/sigreg_b10_s0 \
        --inet_align runs/imagenette/align --inet_epi runs/imagenette/epi --inet_sigreg runs/imagenette/sigreg
"""
import argparse, csv, importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "embedding_analysis" / "data"; OUT.mkdir(exist_ok=True)
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
INET_CLASSES = ["tench", "springer", "cassette", "chainsaw", "church", "horn", "truck", "pump", "golf", "chute"]
ARMS = ["align", "epi", "sigreg"]


def load_module(name, path):
    """Both chassis have a train.py; import each by path under its own name."""
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def tsne(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(X).astype(np.float32)


def pca(X, k=10):
    """Scores on the first k components of centred raw features, and the full explained-variance ratio."""
    m = PCA(min(k, X.shape[1])).fit(X - X.mean(0))
    full = PCA(min(X.shape)).fit(X - X.mean(0)).explained_variance_ratio_
    return m.transform(X - X.mean(0)).astype(np.float32), full.astype(np.float32)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "space", "x", "y", "label", "class"]); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="data/cifar-10-python.tar.gz"); p.add_argument("--data", default="data/imagenette2-160")
    for a in ARMS: p.add_argument(f"--cifar_{a}"); p.add_argument(f"--inet_{a}")
    a = p.parse_args(); dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

    if all(getattr(a, f"cifar_{x}") for x in ARMS):
        T = load_module("cifar_train", ROOT / "cifar" / "train.py")
        data, labels, _ = T.load_data(a.archive); order = np.random.default_rng(746).permutation(50000)
        vx, vy = data[order[45000:]], labels[order[45000:]]
        out, rows, prows = dict(labels=vy.astype(np.int16), class_names=np.array(CIFAR_CLASSES)), [], []
        for arm in ARMS:
            m = T.ResNet18(bn=arm != "sigreg").to(dev)
            m.load_state_dict(torch.load(Path(getattr(a, f"cifar_{arm}")) / "checkpoint.pt", map_location=dev, weights_only=False)["model"]); m.eval()
            h, z = T.extract(m, vx, dev)
            out[f"{arm}_embedding"], out[f"{arm}_backbone"] = z.astype(np.float16), h.astype(np.float16)
            for space, X in [("embedding", z), ("backbone", h)]:
                Y = tsne(X); out[f"{arm}_tsne_{space}"] = Y
                rows += [(arm, space, f"{x:.4f}", f"{y:.4f}", int(l), CIFAR_CLASSES[l]) for (x, y), l in zip(Y, vy)]
                P_, var = pca(X); out[f"{arm}_pca_{space}"], out[f"{arm}_pca_var_{space}"] = P_, var
                prows += [(arm, space, f"{x:.4f}", f"{y:.4f}", int(l), CIFAR_CLASSES[l]) for (x, y), l in zip(P_[:, :2], vy)]
            print(f"cifar {arm}: embedding {z.shape} backbone {h.shape}", flush=True)
        np.savez_compressed(OUT / "cifar_embeddings.npz", **out); write_csv(OUT / "cifar_tsne.csv", rows); write_csv(OUT / "cifar_pca.csv", prows)

    if all(getattr(a, f"inet_{x}") for x in ARMS):
        I = load_module("imagenette_train", ROOT / "imagenette" / "train.py"); ViTEncoder, Views = I.ViTEncoder, I.Views
        val = DataLoader(Views(a.data, "val", V=1), batch_size=128, num_workers=4)
        out, rows, prows, ys = dict(class_names=np.array(INET_CLASSES)), [], [], None
        for arm in ARMS:
            net = ViTEncoder(proj_dim=16).cuda()
            net.load_state_dict(torch.load(Path(getattr(a, f"inet_{arm}")) / "checkpoint.pt", map_location=dev, weights_only=False)["net"]); net.eval()
            PJ, EB, Y = [], [], []
            with torch.no_grad():
                for vs, y in val:
                    with torch.autocast("cuda", dtype=torch.bfloat16): emb, pr = net(vs.cuda())
                    EB.append(emb.float().cpu().numpy()); PJ.append(pr[0].float().cpu().numpy()); Y.append(y.numpy())
            emb, pr, ys = np.concatenate(EB), np.concatenate(PJ), np.concatenate(Y)
            out[f"{arm}_projector"], out[f"{arm}_backbone"] = pr.astype(np.float16), emb.astype(np.float16)
            Y2 = tsne(emb); out[f"{arm}_tsne_backbone"] = Y2
            rows += [(arm, "backbone", f"{x:.4f}", f"{y:.4f}", int(l), INET_CLASSES[l]) for (x, y), l in zip(Y2, ys)]
            for space, X in [("projector", pr), ("backbone", emb)]:
                P_, var = pca(X); out[f"{arm}_pca_{space}"], out[f"{arm}_pca_var_{space}"] = P_, var
                prows += [(arm, space, f"{x:.4f}", f"{y:.4f}", int(l), INET_CLASSES[l]) for (x, y), l in zip(P_[:, :2], ys)]
            print(f"imagenette {arm}: projector {pr.shape} backbone {emb.shape}", flush=True)
            del net; torch.cuda.empty_cache()
        out["labels"] = ys.astype(np.int16)
        np.savez_compressed(OUT / "imagenette_embeddings.npz", **out); write_csv(OUT / "imagenette_tsne.csv", rows); write_csv(OUT / "imagenette_pca.csv", prows)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
