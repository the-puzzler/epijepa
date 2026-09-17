"""Imagenette: the LeJEPA minimal example (MINIMAL.md, galilai-group/lejepa @ c293d29) with one switch.

SIGReg, ViTEncoder, augmentations, optimiser, schedule and online probe are the authors' code
verbatim. The only change is which regulariser sits in their loss:

    loss = regulariser * lamb  +  invariance * (1 - lamb)

    sigreg  their SIGReg statistic                  (replication)
    align   0                                       (collapse control)
    epi     -epiplexity / S0                        (ours)

S0 is calibrated once on four initial batches. Backbone runs in bfloat16 as in their recipe;
the epiplexity score is computed in float64 from the projector output.
"""
import argparse, json, sys, time
from pathlib import Path

import timm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.ops import MLP
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from epiplexity import Reservoir, epiplexity  # noqa: E402


# ---------------------------------------------------------------- theirs, verbatim
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch8_224", pretrained=False, num_classes=512,
                                          drop_path_rate=0.1, img_size=128)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


NORM = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
AUG = v2.Compose([
    v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
    v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    v2.RandomGrayscale(p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
    v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
    v2.RandomHorizontalFlip(),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True), NORM,
])
TEST = v2.Compose([v2.Resize(128), v2.CenterCrop(128), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), NORM])
# ---------------------------------------------------------------- end theirs


class Views(torch.utils.data.Dataset):
    """Their HFDataset contract on a local Imagenette folder (train/ and val/ class subfolders)."""

    def __init__(self, root, split, V=1):
        self.V, self.ds, self.tf = V, ImageFolder(Path(root) / split), AUG if V > 1 else TEST

    def __getitem__(self, i):
        img, y = self.ds[i]
        return torch.stack([self.tf(img.convert("RGB")) for _ in range(self.V)]), y

    def __len__(self):
        return len(self.ds)


RESERVOIR_CHANNELS = (16, 32, 64, 64, 128, 128)  # 128px input -> 4x4 map before pooling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--data", required=True)
    p.add_argument("--method", choices=["sigreg", "align", "epi"], default="sigreg")
    p.add_argument("--lamb", type=float, default=0.02, help="theirs: 0.02; ours needs ~0.1")
    p.add_argument("--V", type=int, default=4); p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=800); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=64, help="reservoir output width")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--grad_ckpt", action="store_true")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck, res = out / "checkpoint.pt", out / "result.json"
    if res.exists():
        print("done already"); return
    torch.manual_seed(a.seed)
    train_ds, test_ds = Views(a.data, "train", V=a.V), Views(a.data, "val", V=1)
    train = DataLoader(train_ds, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    test = DataLoader(test_ds, batch_size=256, num_workers=4, pin_memory=True, persistent_workers=True)
    net = ViTEncoder(proj_dim=a.proj_dim).cuda()
    if a.grad_ckpt: net.backbone.set_grad_checkpointing(True)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).cuda()
    sigreg = SIGReg().cuda()
    reservoir = Reservoir(a.width, 4000 + a.seed, RESERVOIR_CHANNELS).cuda().eval() if a.method == "epi" else None
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": a.lr, "weight_decay": 5e-2},
                             {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}])
    warmup, total = len(train), len(train) * a.epochs
    sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.01, total_iters=warmup),
                               CosineAnnealingLR(opt, T_max=total - warmup, eta_min=1e-3)], milestones=[warmup])
    scaler = GradScaler()

    def views_score(proj, vs):
        return torch.stack([epiplexity(proj[v].float(), vs[:, v].float(), reservoir) for v in range(a.V)]).mean()

    s0, start, hist = 1., 0, []
    if a.method == "epi":  # one-off scale calibration, same role as their loss normalisation
        with torch.no_grad():
            vals = []
            for i, (vs, _) in enumerate(train):
                vs = vs.cuda()
                with autocast("cuda", dtype=torch.bfloat16): _, pr = net(vs)
                vals.append(float(views_score(pr, vs)))
                if i >= 3: break
            s0 = max(sum(vals) / len(vals), 1e-8)
        print(f"S0={s0:.2f}", flush=True)
    if ck.exists():
        st = torch.load(ck, map_location="cuda", weights_only=False)
        net.load_state_dict(st["net"]); probe.load_state_dict(st["probe"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start, hist, s0 = st["epoch"], st["hist"], st["s0"]
        print(f"resumed at epoch {start}", flush=True)

    for epoch in range(start, a.epochs):
        net.train(); probe.train(); t0 = time.perf_counter(); agg = {}
        for vs, y in train:
            vs, y = vs.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                if a.method == "sigreg": reg = sigreg(proj)
                elif a.method == "epi": reg = -views_score(proj, vs) / s0
                else: reg = inv_loss.new_zeros(())
                core = reg * a.lamb + inv_loss * (1 - a.lamb)
                probe_loss = F.cross_entropy(probe(emb.detach()), y.repeat_interleave(a.V))
                loss = core + probe_loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            for k, v in [("core", core), ("inv", inv_loss), ("reg", reg), ("probe", probe_loss)]:
                agg[k] = agg.get(k, 0.) + float(v.detach())
        log = {k: v / len(train) for k, v in agg.items()}
        net.eval(); probe.eval(); correct = 0
        with torch.inference_mode():
            for vs, y in test:
                with autocast("cuda", dtype=torch.bfloat16):
                    correct += (probe(net(vs.cuda(non_blocking=True))[0]).argmax(1) == y.cuda()).sum().item()
        log.update(epoch=epoch + 1, test_acc=correct / len(test_ds), seconds=time.perf_counter() - t0)
        hist.append(log)
        torch.save(dict(net=net.state_dict(), probe=probe.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch + 1, hist=hist, s0=s0), ck)
        (out / "history.json").write_text(json.dumps(hist))
        print(f"{a.method} lamb={a.lamb:g} seed={a.seed}: {epoch + 1}/{a.epochs} acc={log['test_acc'] * 100:.2f}% "
              f"inv={log['inv']:.4f} reg={log['reg']:.4f} {log['seconds']:.0f}s", flush=True)
    best = max(h["test_acc"] for h in hist)
    res.write_text(json.dumps(dict(method=a.method, lamb=a.lamb, seed=a.seed, final_acc=hist[-1]["test_acc"], best_acc=best,
                                   epochs=a.epochs, history=hist), indent=1))
    print(f"FINAL {a.method} lamb={a.lamb:g} seed={a.seed}: final {hist[-1]['test_acc'] * 100:.2f}%  best {best * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
