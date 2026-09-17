"""CIFAR-10: ResNet-18, 64-d embedding, two augmented views.

    loss = alignment / A0  +  beta * regulariser

    align   regulariser = 0                     (collapse control)
    epi     regulariser = -epiplexity / S0      (ours: maximise the score)
    sigreg  regulariser =  SIGReg / SIGReg0     (LeJEPA reference, authors' package)

A0, S0, SIGReg0 are calibrated once on four initial batches so beta is on a common scale.
Runs every (method, beta, seed) in the config, selects beta per method by mean validation
backbone accuracy, and reports the official test split only for the selected betas.
Float32, no TF32/AMP: the log-det is numerically delicate.
"""
import argparse, copy, hashlib, io, json, pickle, sys, tarfile, time, urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from epiplexity import Reservoir, epiplexity  # noqa: E402

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
METHODS = ["align", "epi", "sigreg"]


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, inc, c, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(inc, c, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(c)
        self.skip = nn.Identity() if inc == c and stride == 1 else nn.Sequential(
            nn.Conv2d(inc, c, 1, stride, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ResNet18(nn.Module):
    """CIFAR stem. Output BatchNorm (non-affine) for align/epi; raw output for sigreg, whose
    target already fixes the scale."""

    def __init__(self, dim=64, bn=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, inc = [], 64
        for c in [64, 128, 256, 512]:
            layers += [Block(inc, c, 1 if inc == c else 2), Block(c, c)]; inc = c
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(512, dim)
        self.norm = nn.BatchNorm1d(dim, affine=False) if bn else nn.Identity()

    def features(self, x):
        return self.blocks(self.stem(x)).mean((2, 3))

    def forward(self, x):
        return self.norm(self.head(self.features(x)))


def augment(x, g):
    """Crop 65-100% side, flip, brightness/contrast, 20% grayscale. x in [0,1] -> [-1,1]."""
    n, dev = x.shape[0], x.device
    scale = .65 + .35 * torch.rand(n, generator=g, device=dev)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = scale * torch.where(torch.rand(n, generator=g, device=dev) < .5, -1., 1.)
    theta[:, 1, 1] = scale
    theta[:, :, 2] = (torch.rand(n, 2, generator=g, device=dev) * 2 - 1) * (1 - scale[:, None])
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False, padding_mode="reflection")
    mean = x.mean((2, 3), keepdim=True)
    contrast = .8 + .4 * torch.rand(n, 1, 1, 1, generator=g, device=dev)
    bright = .8 + .4 * torch.rand(n, 1, 1, 1, generator=g, device=dev)
    x = ((x - mean) * contrast + mean) * bright
    x = torch.where(torch.rand(n, 1, 1, 1, generator=g, device=dev) < .2, x.mean(1, keepdim=True), x)
    return (x.clamp(0, 1) - .5) / .5


def make_sigreg(device):
    import lejepa
    return lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=lejepa.univariate.EppsPulley(n_points=17), num_slices=1024).to(device)


# ----------------------------------------------------------------------------- data
def load_data(path, include_test=False):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CIFAR_URL, p)
    payload = p.read_bytes()
    if hashlib.md5(payload).hexdigest() != CIFAR_MD5:
        raise ValueError("CIFAR archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(payload)) as tf:
        batches = [pickle.load(tf.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="bytes") for i in range(1, 6)]
        data = np.concatenate([b[b"data"] for b in batches]); labels = np.concatenate([np.array(b[b"labels"]) for b in batches])
        test = None
        if include_test:
            b = pickle.load(tf.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
            test = (torch.from_numpy(b[b"data"].reshape(-1, 3, 32, 32)), np.array(b[b"labels"]))
    return torch.from_numpy(data.reshape(-1, 3, 32, 32)), labels, test


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def extract(model, images, device, batch=128):
    model.eval(); hh, zz = [], []
    for x in images.split(batch):
        h = model.features((x.to(device).float() / 255 - .5) / .5)
        hh.append(h.cpu().numpy()); zz.append(model.norm(model.head(h)).cpu().numpy())
    return np.concatenate(hh), np.concatenate(zz)


def effective_rank(z):
    c = z.astype(np.float64) - z.mean(0)
    e = np.linalg.eigvalsh(c.T @ c / max(len(z) - 1, 1)).clip(0); p = e / max(e.sum(), 1e-30)
    return float(np.exp(-(p * np.log(p + 1e-30)).sum())) if e.sum() > 1e-20 else 0.


@torch.no_grad()
def evaluate(model, tx, ty, vx, vy, cfg, test=None):
    """Ridge probes (alpha 1) fit on training features; validation and, if given, test accuracy."""
    h, z = extract(model, tx[: cfg["probe_size"]], cfg["device"])
    hv, zv = extract(model, vx, cfg["device"])
    ht_zt = extract(model, test[0], cfg["device"]) if test is not None else None
    out = {}
    for j, (name, a, b) in enumerate([("backbone", h, hv), ("embedding", z, zv)]):
        mu = a.mean(0); scale = max(float(np.std(a - mu)), 1e-12)
        probe = RidgeClassifier(alpha=1.).fit((a - mu) / scale, ty[: len(a)])
        out[name] = dict(validation_accuracy=float(probe.score((b - mu) / scale, vy)), effective_rank=effective_rank(b))
        if ht_zt is not None:
            out[name]["test_accuracy"] = float(probe.score((ht_zt[j] - mu) / scale, test[1]))
    return out


# ----------------------------------------------------------------------------- train
def init(seed, method, cfg):
    torch.manual_seed(seed)
    model = ResNet18(bn=method != "sigreg").to(cfg["device"])
    reservoir = Reservoir(cfg["width"], 4000 + seed).to(cfg["device"]).eval()
    return model, reservoir


def calibrate(model, method, reservoir, sig, tx, cfg, seed):
    """Loss scales A0 / S0 / SIGReg0 from the untrained encoder on four batches, using batch statistics."""
    frozen = copy.deepcopy(model).requires_grad_(False).train()
    for m in frozen.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum, m.track_running_stats = 0., False
    g = torch.Generator(device=cfg["device"]).manual_seed(2000 + seed)
    aa, rr = [], []
    with torch.no_grad():
        for i in range(4):
            idx = torch.randperm(len(tx), generator=torch.Generator().manual_seed(2900 + i))[: cfg["batch"]]
            raw = tx[idx].to(cfg["device"]).float() / 255
            a, b = augment(raw, g), augment(raw, g); za, zb = frozen(a), frozen(b)
            aa.append(float((za - zb).square().mean() / 4))
            if method == "epi":
                rr.append(float((epiplexity(za, a, reservoir) + epiplexity(zb, b, reservoir)) / 2))
            elif method == "sigreg":
                rr.append(float((sig(za) + sig(zb)) / 2))
    return max(float(np.mean(aa)), 1e-8), (max(float(np.mean(rr)), 1e-8) if rr else 1.)


def train_one(cfg, method, beta, seed, tx, ty, vx, vy):
    dev = cfg["device"]
    folder = Path(cfg["out"]) / f"{method}_b{beta:g}_s{seed}"; folder.mkdir(parents=True, exist_ok=True)
    result_path, ckpt_path = folder / "result.json", folder / "checkpoint.pt"
    if result_path.exists():
        return json.loads(result_path.read_text())
    model, reservoir = init(seed, method, cfg)
    sig = make_sigreg(dev) if method == "sigreg" else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    g = torch.Generator(device=dev).manual_seed(1000 + seed)
    align0, reg0 = calibrate(model, method, reservoir, sig, tx, cfg, seed)
    history, start, seconds = [], 0, 0.
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if sig is not None: sig.load_state_dict(ck["sigreg"])
        history, start, seconds, align0, reg0 = ck["history"], ck["epoch"], ck["seconds"], ck["align0"], ck["reg0"]
        g.set_state(ck["aug_rng"].cpu()); torch.set_rng_state(ck["cpu_rng"].cpu())
        if dev.startswith("cuda"): torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
    for epoch in range(start, cfg["epochs"]):
        model.train(); t0 = time.perf_counter(); tot = np.zeros(4); n = 0
        order = torch.randperm(len(tx), generator=torch.Generator().manual_seed(seed * 10000 + epoch))
        for i in range(0, len(tx) - cfg["batch"] + 1, cfg["batch"]):
            raw = tx[order[i: i + cfg["batch"]]].to(dev).float() / 255
            x1, x2 = augment(raw, g), augment(raw, g); z1, z2 = model(x1), model(x2)
            align = (z1 - z2).square().mean() / 4 / align0
            if method == "epi":
                s = (epiplexity(z1, x1, reservoir) + epiplexity(z2, x2, reservoir)) / 2
                reg, ratio = -s / reg0, float(s.detach() / reg0)
            elif method == "sigreg":
                reg, ratio = (sig(z1) + sig(z2)) / (2 * reg0), 0.
            else:
                reg, ratio = align.new_zeros(()), 0.
            loss = align + beta * reg
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {method} beta={beta} seed={seed} epoch={epoch}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += [float(loss.detach()), float(align.detach()), float(reg.detach()), ratio]; n += 1
        if dev.startswith("cuda"): torch.cuda.synchronize()
        el = time.perf_counter() - t0; seconds += el
        log = dict(zip(["loss", "alignment", "regulariser", "score_ratio"], (tot / max(n, 1)).tolist()), epoch=epoch + 1, seconds=el)
        if (epoch + 1) % cfg["diagnostic_every"] == 0 or epoch + 1 == cfg["epochs"]:
            log["embedding_rank"] = effective_rank(extract(model, vx[:2048], dev)[1])
        history.append(log)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), sigreg=sig.state_dict() if sig else None,
                        epoch=epoch + 1, history=history, seconds=seconds, align0=align0, reg0=reg0, aug_rng=g.get_state(),
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if dev.startswith("cuda") else None), ckpt_path)
        print(f"{method} beta={beta:g} seed={seed}: {epoch + 1}/{cfg['epochs']} loss={log['loss']:.4f} align={log['alignment']:.4f} {el:.0f}s", flush=True)
    row = dict(method=method, beta=beta, seed=seed, train_hours=seconds / 3600, metrics=evaluate(model, tx, ty, vx, vy, cfg),
               checkpoint=str(ckpt_path), history=history)
    result_path.write_text(json.dumps(row, indent=1))
    return row


def protocol(cfg, tx, ty, vx, vy, test=None):
    out = Path(cfg["out"]); rows = []
    for seed in cfg["seeds"]:
        for method in METHODS:
            for beta in ([0.] if method == "align" else cfg["weights"]):
                rows.append(train_one(cfg, method, beta, seed, tx, ty, vx, vy))
                (out / "validation_results.json").write_text(json.dumps(rows, indent=1))
    # Pre-declared selector: mean validation backbone accuracy over seeds.
    chosen = {}
    for method in METHODS:
        cands = [0.] if method == "align" else cfg["weights"]
        best = max(cands, key=lambda b: np.mean([r["metrics"]["backbone"]["validation_accuracy"] for r in rows if r["method"] == method and r["beta"] == b]))
        chosen[method] = best
    (out / "selection.json").write_text(json.dumps(chosen, indent=1))
    if test is not None:
        trows = []
        for method, beta in chosen.items():
            for seed in cfg["seeds"]:
                model, _ = init(seed, method, cfg)
                model.load_state_dict(torch.load(out / f"{method}_b{beta:g}_s{seed}" / "checkpoint.pt", map_location=cfg["device"], weights_only=False)["model"])
                trows.append(dict(method=method, beta=beta, seed=seed, metrics=evaluate(model, tx, ty, vx, vy, cfg, test=test)))
        (out / "test_results.json").write_text(json.dumps(trows, indent=1))
    return rows, chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--archive", default="data/cifar-10-python.tar.gz")
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch", type=int, default=256)
    p.add_argument("--width", type=int, default=64, help="reservoir output width")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--weights", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1., 3., 10.])
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--diagnostic_every", type=int, default=10)
    p.add_argument("--train_size", type=int, default=45000); p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--probe_size", type=int, default=45000)
    a = p.parse_args()
    cfg = dict(vars(a), device="cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    Path(cfg["out"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["out"]) / "config.json").write_text(json.dumps(cfg, indent=1))
    data, labels, test = load_data(cfg["archive"], include_test=a.stage == "test")
    order = np.random.default_rng(746).permutation(50000)
    ti, vi = order[:45000][: cfg["train_size"]], order[45000:][: cfg["val_size"]]
    _, chosen = protocol(cfg, data[ti], labels[ti], data[vi], labels[vi], test)
    print("selected weights:", json.dumps(chosen))


if __name__ == "__main__":
    main()
