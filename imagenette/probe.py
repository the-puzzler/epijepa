"""Frozen-backbone probe for Imagenette runs.

The number the training script reports comes from a head trained jointly with the encoder.
This refits a clean logistic-regression probe and a 20-NN classifier on extracted features
of the frozen backbone, and reports its effective rank. Use it for the comparison.

    python imagenette/probe.py --data data/imagenette2-160 runs/imagenette/sigreg runs/imagenette/epi_l0.1
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ViTEncoder, Views  # noqa: E402


@torch.no_grad()
def features(net, loader):
    X, Y = [], []
    for vs, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = net.backbone(vs.cuda(non_blocking=True).flatten(0, 1))
        X.append(e.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)


def effective_rank(X):
    c = X - X.mean(0); e = np.linalg.eigvalsh(np.cov(c.T)).clip(0); p = e / max(e.sum(), 1e-30)
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+"); p.add_argument("--data", required=True); p.add_argument("--proj_dim", type=int, default=16)
    a = p.parse_args()
    tr = DataLoader(Views(a.data, "train", V=1), batch_size=64, num_workers=4)
    va = DataLoader(Views(a.data, "val", V=1), batch_size=64, num_workers=4)
    for run in a.runs:
        run = Path(run)
        net = ViTEncoder(proj_dim=a.proj_dim).cuda()
        net.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=False)["net"]); net.eval()
        Xt, Yt = features(net, tr); Xv, Yv = features(net, va)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Zt, Zv = (Xt - mu) / sd, (Xv - mu) / sd
        out = dict(offline_linear=LogisticRegression(max_iter=2000, n_jobs=-1).fit(Zt, Yt).score(Zv, Yv) * 100,
                   knn20=KNeighborsClassifier(20).fit(Zt, Yt).score(Zv, Yv) * 100,
                   backbone_rank=effective_rank(Zt))
        (run / "probe.json").write_text(json.dumps(out, indent=1))
        r = json.loads((run / "result.json").read_text())
        print(f"{run.name:24s} online {r['final_acc'] * 100:6.2f}%  offline-linear {out['offline_linear']:6.2f}%  "
              f"knn20 {out['knn20']:6.2f}%  backbone rank {out['backbone_rank']:5.1f}")
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
