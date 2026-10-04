"""Train LeWM with either SIGReg (LeWM's anti-collapse term) or epiplexity (EpiWM; the score is from EpiJEPA) as the regulariser.

Everything except the regulariser is LeWM's own recipe: the released checkpoints' architecture (ViT-tiny encoder,
AdaLN predictor, MLP projectors; config/train/model/*.yaml), the dataset and transforms (utils.py, verbatim from
LeWM), AdamW lr 5e-5 / wd 1e-3, 1% linear warmup + cosine decay, grad-clip 1.0, bf16 autocast, batch 128,
history 3, frameskip 5. LeWM trains with Lightning; this is the same loop in plain PyTorch so the loss is visible
every few steps. Checkpoints are written with stable_worldmodel's save_pretrained, so LeWM's eval.py loads them.

  SIGReg:  loss = pred + 0.09 * SIGReg(z)                                   (model=sigreg)
  Epi:     loss = pred - lambda * S(z) / S0                                  (model=epi)
           S = epiplexity of the embeddings w.r.t. a frozen random CNN reservoir of the same frames
           (../epiplexity.py); S0 = S of the untrained model, averaged over 4 batches. The Epi arm's only
           architectural change is a non-affine BatchNorm on the projector output (module.ProjectorBN).

Settings behind the reported results (steps matter: the cosine schedule spans max_steps):
  python train.py data=tworoom model=epi loss.reg=epi loss.epi.weight=0.03 max_steps=30000  run_name=tw_epi
  python train.py data=pusht   model=epi loss.reg=epi loss.epi.weight=0.1  max_steps=60000  run_name=pusht_epi
  python train.py data=ogb     model=epi loss.reg=epi loss.epi.weight=0.03 max_steps=60000  run_name=cube_epi
  python train.py data=dmc     model=epi loss.reg=epi loss.epi.weight=0.3  max_steps=200000 run_name=reacher_epi
  python train.py data=pusht   model=sigreg loss.reg=sigreg max_steps=60000 run_name=pusht_sigreg   # our SIGReg control
Any config key can be overridden with key=value (e.g. seed=2, num_workers=4).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from stable_worldmodel.wm.utils import save_pretrained

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from epiplexity import Reservoir, epiplexity  # noqa: E402  (EpiJEPA, repo root)
from module import SIGReg  # noqa: E402                     (LeWM, verbatim)
from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402  (LeWM, verbatim)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_cfg(argv):
    """LeWM's config/train/lewm.yaml + data/<data>.yaml + model/<model>.yaml, then key=value overrides."""
    cli = dict(a.lstrip("+").split("=", 1) for a in argv)
    raw = OmegaConf.to_container(OmegaConf.load(HERE / "config/train/lewm.yaml"), resolve=False)
    for k in ("defaults", "subdir", "trainer", "wandb", "output_model_name"):  # Lightning/hydra-launcher only
        raw.pop(k, None)
    cfg = OmegaConf.create(raw)
    cfg.data = OmegaConf.load(HERE / f"config/train/data/{cli.pop('data', 'pusht')}.yaml")
    cfg.model = OmegaConf.load(HERE / f"config/train/model/{cli.pop('model', 'sigreg')}.yaml")
    cfg.run_name, cfg.max_steps, cfg.ckpt_every, cfg.log_every = "unnamed", 60000, 10000, 200
    cfg.out_dir = "runs"
    for k in cli:
        if OmegaConf.select(cfg, k) is None:
            raise KeyError(f"unknown config key {k!r}")
    cfg.merge_with(OmegaConf.from_dotlist([f"{k}={v}" for k, v in cli.items()]))
    return cfg


def effective_rank(z):
    sv = torch.linalg.svdvals((z - z.mean(0)).float())
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def main():
    cfg = load_cfg(sys.argv[1:])
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = "cuda"
    use_epi = cfg.loss.reg == "epi"
    assert cfg.loss.reg in ("sigreg", "epi")
    assert use_epi == cfg.model.projector["_target_"].endswith("ProjectorBN"), "use model=epi with loss.reg=epi"

    # ---- data (as LeWM's train.py)
    dcfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dcfg.pop("name")
    if name.endswith(".lance") and not (Path(os.environ["STABLEWM_HOME"]) / "datasets" / name).exists():
        name = name.replace(".lance", ".h5")  # the HF mirror ships HDF5
    dataset = swm.data.load_dataset(name, transform=None, **dcfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    transforms += [get_column_normalizer(dataset, c, c) for c in cfg.data.dataset.keys_to_load if not c.startswith("pixels")]
    dataset.transform = spt.data.transforms.Compose(*transforms)
    cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=gen)
    lk = dict(batch_size=cfg.loader.batch_size, num_workers=cfg.num_workers, pin_memory=True,
              persistent_workers=True, prefetch_factor=cfg.loader.prefetch_factor)
    train_dl = torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=True, generator=gen, **lk)
    val_dl = torch.utils.data.DataLoader(val_set, shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(0), **lk)

    # ---- model, regularisers, optimiser
    model = hydra.utils.instantiate(cfg.model).to(dev)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(dev)
    reservoir = Reservoir(cfg.loss.epi.width, cfg.loss.epi.reservoir_seed).to(dev).eval()  # frozen random CNN
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    T, warm = cfg.max_steps, max(1, int(0.01 * cfg.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, T - warm)))))
    H, P = cfg.history_size, cfg.num_preds

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"[{cfg.run_name}] reg={cfg.loss.reg} steps={T} train={len(train_set)} val={len(val_set)} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    def forward(batch):
        """Prediction loss, SIGReg (always computed; logged for both arms), embeddings and frames."""
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.autocast("cuda", torch.bfloat16):
            out = model.encode(batch)
            emb, act = out["emb"], out["act_emb"]
            pred = model.predict(emb[:, :H], act[:, :H])
        emb = emb.float()
        l_pred = (pred.float() - emb[:, P:]).pow(2).mean()
        l_sig = sigreg(emb.transpose(0, 1))
        flat = emb.reshape(-1, emb.size(-1))
        pix = batch["pixels"].reshape(-1, *batch["pixels"].shape[2:]).float()
        return l_pred, l_sig, flat, pix

    # ---- S0: epiplexity of the untrained model on four training batches (EpiJEPA's normaliser)
    model.train()
    it = iter(train_dl)
    with torch.no_grad():
        s0 = []
        for _ in range(4):
            _, _, flat, pix = forward(next(it))
            s0.append(float(epiplexity(flat, pix, reservoir)))  # fp32 here; the per-step score runs under bf16 autocast
    S0 = float(np.mean(s0))
    print(f"[{cfg.run_name}] S0={S0:.3f}", flush=True)

    log, t0, step, epoch = [], time.time(), 0, 0
    while step < T:
        for batch in train_dl:
            l_pred, l_sig, flat, pix = forward(batch)
            if use_epi:
                with torch.autocast("cuda", torch.bfloat16):
                    s = epiplexity(flat, pix, reservoir).float() / S0
                loss = l_pred - cfg.loss.epi.weight * s
            else:
                loss = l_pred + cfg.loss.sigreg.weight * l_sig
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            if step % cfg.log_every == 0 or step == 1:
                with torch.no_grad():
                    epi_ratio = float(s) if use_epi else float(epiplexity(flat.detach(), pix, reservoir)) / S0
                row = dict(step=step, epoch=epoch, loss=loss.item(), pred=l_pred.item(), sigreg=l_sig.item(),
                           epi_ratio=epi_ratio, erank=effective_rank(flat.detach()), grad_norm=float(gn),
                           lr=sched.get_last_lr()[0], it_s=step / (time.time() - t0))
                log.append(row)
                print(f"[{cfg.run_name}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

            if step % cfg.ckpt_every == 0 or step == T:
                model.eval()
                with torch.no_grad():  # 20 validation batches
                    vals = []
                    for i, vb in enumerate(val_dl):
                        vp, vs, vflat, vpix = forward(vb)
                        vals.append((float(vp), float(vs), float(epiplexity(vflat, vpix, reservoir)) / S0, effective_rank(vflat)))
                        if i == 19:
                            break
                v = np.mean(vals, 0)
                val = dict(step=step, val_pred=v[0], val_sigreg=v[1], val_epi_ratio=v[2], val_erank=v[3])
                print(f"[{cfg.run_name}] VAL " + " ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in val.items()), flush=True)
                log.append(val)
                save_pretrained(model, run_name=cfg.run_name, config=cfg.model, filename=f"weights_step{step}.pt")
                json.dump(dict(S0=S0, log=log), open(out_dir / "train_log.json", "w"))
                model.train()
            if step >= T:
                break
        epoch += 1
    print(f"[{cfg.run_name}] done in {(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
