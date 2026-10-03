"""Shared bits for the analysis scripts: per-environment datasets, state labels and the checkpoints compared.

Labels are the parts of the true state that are visible in the frame (TwoRoom's target and Reacher's target are
not rendered, so they are left out; the cube's yaw is only defined up to its 90-degree symmetry).

Checkpoints are resolved by stable_worldmodel's load_pretrained relative to $STABLEWM_HOME/checkpoints:
  ours      the best EpiJEPA checkpoint per environment (the ones uploaded to Hugging Face)
  released  the official LeWM checkpoint (HF quentinll/lewm-*, keys renamed with ../convert_hf4_ckpt.py)
"""
import os

import h5py
import numpy as np
import torch

DATA = os.path.join(os.environ.get("STABLEWM_HOME", "."), "datasets")

ENVS = {
    "tworoom": dict(h5="tworoom.h5", ep="ep_idx",
                    labels={"agent_x": ("pos_agent", 0), "agent_y": ("pos_agent", 1)},
                    ours="tw_epi_0.03/weights_step30000.pt", sigreg="tw_sigreg_0.09/weights_step30000.pt",
                    released="official_tworooms_v5"),
    "pusht": dict(h5="pusht_expert_train.h5", ep="episode_idx",
                  labels={"agent_x": ("state", 0), "agent_y": ("state", 1), "block_x": ("state", 2),
                          "block_y": ("state", 3), "block_angle": ("state", 4)},
                  ours="pusht_epi_0.1_60k_seed3/weights_step60000.pt", sigreg="pusht_sigreg_0.09_60k/weights_step60000.pt",
                  released="official_pusht_v5"),
    "cube": dict(h5="ogbench/cube_single_expert.h5", ep="ep_idx",
                 labels={"block_x": ("privileged_block_0_pos", 0), "block_y": ("privileged_block_0_pos", 1),
                         "block_z": ("privileged_block_0_pos", 2), "block_yaw": ("privileged_block_0_yaw", 0),
                         "effector_x": ("proprio_effector_pos", 0), "effector_y": ("proprio_effector_pos", 1),
                         "effector_z": ("proprio_effector_pos", 2), "gripper_opening": ("proprio_gripper_opening", 0)},
                 ours="cube_epi_0.03_seed3/weights_step60000.pt", sigreg="cube_sigreg_0.09/weights_step60000.pt",
                 released="official_cube_v5"),
    "reacher": dict(h5="reacher.h5", ep="ep_idx",
                    labels={"finger_x": ("finger_pos", 0), "finger_y": ("finger_pos", 1),
                            "joint_0": ("qpos", 0), "joint_1": ("qpos", 1)},
                    ours="rch_epi_0.3_200k/weights_step200000.pt", sigreg="rch_sigreg_0.09/weights_step60000.pt",
                    released="official_reacher_v5"),
}

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sample_rows(env, n, seed=0, offsets=(0,)):
    """n random dataset rows r such that r + max(offsets) stays inside the same episode. Returns sorted rows."""
    with h5py.File(os.path.join(DATA, ENVS[env]["h5"]), "r") as f:
        ep = f[ENVS[env]["ep"]][:]
    rng = np.random.default_rng(seed)
    k = max(offsets)
    ok = np.nonzero(ep[: len(ep) - k] == ep[k:])[0] if k else np.arange(len(ep))
    return np.sort(rng.choice(ok, n, replace=False))


def load_rows(env, rows):
    """Pixels (uint8 [N,224,224,3]), labels (dict name -> [N]) and episode / step indices for the given rows."""
    spec = ENVS[env]
    with h5py.File(os.path.join(DATA, spec["h5"]), "r") as f:
        pix = f["pixels"][rows]
        labels = {}
        for name, (col, j) in spec["labels"].items():
            v = f[col][rows]
            labels[name] = v if j is None else v[:, j]
        labels["episode"] = f[spec["ep"]][rows]
        labels["step"] = f["step_idx"][rows]
    return pix, labels


def to_input(pix):
    """uint8 HWC -> ImageNet-normalised float NCHW, the same maths as LeWM's get_img_preprocessor (utils.py)."""
    x = torch.as_tensor(pix).permute(0, 3, 1, 2).float() / 255
    return (x - IMNET_MEAN) / IMNET_STD


@torch.no_grad()
def encode(model, pix, bs=256):
    """Embeddings z (projector output, the planning latent) of single frames: [N, 192] float64 on CPU."""
    out = []
    for i in range(0, len(pix), bs):
        x = to_input(pix[i:i + bs]).cuda()[:, None]
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model.encode({"pixels": x})["emb"][:, 0].float().cpu())
    return torch.cat(out).double()
