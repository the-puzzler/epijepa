"""Representation probes, identical for both models (our best EpiWM checkpoint and the released LeWM one):
  linear_r2 / mlp_r2   how well the true state is decodable from the latent z_t (ridge / 2-layer MLP; 2000 train,
                       400 test windows). Angles are probed as (sin, cos) (cube yaw modulo its 90-degree symmetry).
  rollout_err          open-loop rollout with the true actions for 5 steps; the state decoded from the predicted
                       latents vs the true state, mean |error| in state-std units (rollout_floor = same probe on the
                       true latents, i.e. the decoding error alone)
  goal_rank_rho        Spearman correlation between ||z_t - z_goal|| and steps-to-goal along real trajectories
                       (does latent distance order states by how far they are from the goal? CEM plans with it)
  erank / top2_var     effective rank of the latent and variance in its top-2 PCs
usage: python probes.py [env ...]  -> probes/probes.json, probes/probes_summary.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # worldmodel/
from common import ENVS  # noqa: E402
from train import load_cfg  # noqa: E402
from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402

H = 5                       # rollout horizon (steps of frameskip 5)
N_TRAIN, N_TEST = 2000, 400
ANGLES = {"block_angle": 1, "block_yaw": 4, "joint_0": 1, "joint_1": 1}  # name -> symmetry order (cube: 4-fold)
DATA_CFG = {"tworoom": "tworoom", "pusht": "pusht", "cube": "ogb", "reacher": "dmc"}
MODELS = [("epiwm", "ours"), ("released_lewm", "released")]
OUT = os.path.join(HERE, "probes")


def windows(env):
    """N_TRAIN + N_TEST random windows of H+1 frames (frameskip 5): pixels, normalised actions, probe targets."""
    cfg = load_cfg([f"data={DATA_CFG[env]}"])
    d = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = d.pop("name").replace(".lance", ".h5")
    cols = sorted({c for c, _ in ENVS[env]["labels"].values()})
    d.update(num_steps=H + 1, keys_to_load=["pixels", "action", *cols], keys_to_cache=["action", *cols])
    d.pop("keys_to_merge", None)
    ds = swm.data.load_dataset(name, transform=None, **d)
    ds.transform = spt.data.transforms.Compose(get_img_preprocessor("pixels", "pixels", 224),
                                               get_column_normalizer(ds, "action", "action"))
    idx = np.random.default_rng(77).choice(len(ds), N_TRAIN + N_TEST, replace=False)
    items = [ds[int(i)] for i in idx]
    X = torch.stack([it["pixels"] for it in items])                         # [N, H+1, 3, 224, 224]
    A = torch.nan_to_num(torch.stack([torch.as_tensor(it["action"]) for it in items]).float())
    names, targets = [], []
    for n, (c, j) in ENVS[env]["labels"].items():
        v = torch.stack([torch.as_tensor(it[c]) for it in items]).double()
        v = v if v.dim() == 2 else v[..., 0 if j is None else j]           # [N, H+1]
        if n in ANGLES:  # probe (sin, cos) of k * angle, k = rotational symmetry of the object
            k = ANGLES[n]
            names += [f"{n}_sin", f"{n}_cos"]; targets += [torch.sin(k * v), torch.cos(k * v)]
        else:
            names.append(n); targets.append(v)
    return X, A, torch.stack(targets, -1), names                            # targets [N, H+1, k]


def ridge(Z, Y, lam=1e-2):
    Zb = torch.cat([Z, torch.ones(len(Z), 1, dtype=Z.dtype)], 1)
    return torch.linalg.solve(Zb.T @ Zb + lam * len(Zb) * torch.eye(Zb.shape[1], dtype=Zb.dtype), Zb.T @ Y)


def apply(W, Z):
    return torch.cat([Z, torch.ones(*Z.shape[:-1], 1, dtype=Z.dtype)], -1) @ W


def r2(P, Y):
    return (1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).tolist()


def mlp_probe(Ztr, Ytr, Zte, Yte, steps=1500):
    torch.manual_seed(0)
    mu, sd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(Ztr.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256),
                              torch.nn.GELU(), torch.nn.Linear(256, Ytr.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    zt, yt = Ztr.float().cuda(), ((Ytr - mu) / sd).float().cuda()
    for _ in range(steps):
        i = torch.randint(0, len(zt), (256,), device="cuda")
        loss = ((net(zt[i]) - yt[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return r2(net(Zte.float().cuda()).cpu().double() * sd + mu, Yte)


@torch.no_grad()
def latents(model, X, A):
    """True latents of all H+1 frames and open-loop predicted latents for steps 1..H (history up to 3, as trained)."""
    Z = torch.cat([model.encode({"pixels": X[i:i + 32].cuda()})["emb"].float().cpu() for i in range(0, len(X), 32)])
    with torch.autocast("cuda", torch.bfloat16):
        emb, act = Z[:, :1].cuda(), model.action_encoder(A.cuda())
        for _ in range(H):
            lo = max(0, emb.shape[1] - 3)
            emb = torch.cat([emb, model.predict(emb[:, lo:], act[:, lo:emb.shape[1]])[:, -1:].float()], 1)
    return Z.double(), emb[:, 1:].cpu().double()


def probe(model, X, A, Tg, names):
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, None)
    with torch.autocast("cuda", torch.bfloat16):
        Z, Zp = latents(model, X, A)
    Z0, T0 = Z[:, 0], Tg[:, 0]
    lin = r2(apply(ridge(Z0[tr], T0[tr]), Z0[te]), T0[te])
    mlp = mlp_probe(Z0[tr], T0[tr], Z0[te], T0[te])
    sd = Tg[:, 1:].reshape(-1, Tg.shape[-1]).std(0)
    Wall = ridge(Z[tr].reshape(-1, Z.shape[-1]), Tg[tr].reshape(-1, Tg.shape[-1]))
    roll = ((apply(Wall, Zp[te]) - Tg[te, 1:]).abs() / sd).mean(-1).mean(0).tolist()
    floor = ((apply(Wall, Z[te, 1:]) - Tg[te, 1:]).abs() / sd).mean(-1).mean(0).tolist()
    dg = (Z[:, :H] - Z[:, H:H + 1]).norm(dim=-1).numpy()
    rho = float(np.nanmean([spearmanr(dg[n], np.arange(H, 0, -1)).correlation for n in range(len(Z))]))
    Zc = Z0 - Z0.mean(0); sv = torch.linalg.svdvals(Zc); p = sv / sv.sum(); ev = sv ** 2 / (sv ** 2).sum()
    return dict(linear_r2=dict(zip(names, lin)), mlp_r2=dict(zip(names, mlp)),
                linear_r2_mean=float(np.mean(lin)), mlp_r2_mean=float(np.mean(mlp)),
                rollout_err=roll, rollout_floor=floor, goal_rank_rho=rho,
                erank=float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum())), top2_var=float(ev[:2].sum()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/probes.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for env in sys.argv[1:] or list(ENVS):
        X, A, Tg, names = windows(env)
        for tag, key in MODELS:
            model = swm.wm.utils.load_pretrained(ENVS[env][key]).cuda().eval()
            res.setdefault(env, {})[tag] = dict(checkpoint=ENVS[env][key], **probe(model, X, A, Tg, names))
            del model; torch.cuda.empty_cache()
            r = res[env][tag]
            print(f"{env:8s} {tag:14s} linR2 {r['linear_r2_mean']:.3f} mlpR2 {r['mlp_r2_mean']:.3f} "
                  f"roll t1/t5 {r['rollout_err'][0]:.3f}/{r['rollout_err'][-1]:.3f} goal_rho {r['goal_rank_rho']:.3f} "
                  f"erank {r['erank']:.0f}", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    rows = [dict(env=e, model=m, linear_r2=r["linear_r2_mean"], mlp_r2=r["mlp_r2_mean"], rollout_err_t1=r["rollout_err"][0],
                 rollout_err_t5=r["rollout_err"][-1], rollout_floor_t1=r["rollout_floor"][0], goal_rank_rho=r["goal_rank_rho"],
                 erank=r["erank"], top2_var=r["top2_var"]) for e, d in res.items() for m, r in d.items()]
    pd.DataFrame(rows).round(4).to_csv(f"{OUT}/probes_summary.csv", index=False)
    print("saved", path)
