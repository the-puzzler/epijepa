"""Rename HF-transformers-4.x ViT keys in a released LeWM state dict to the transformers-5 layout."""
import re
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")
rules = [(r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query", r"encoder.layers.\1.attention.q_proj"),
         (r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key", r"encoder.layers.\1.attention.k_proj"),
         (r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value", r"encoder.layers.\1.attention.v_proj"),
         (r"encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense", r"encoder.layers.\1.attention.o_proj"),
         (r"encoder\.encoder\.layer\.(\d+)\.intermediate\.dense", r"encoder.layers.\1.mlp.fc1"),
         (r"encoder\.encoder\.layer\.(\d+)\.output\.dense", r"encoder.layers.\1.mlp.fc2"),
         (r"encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1.")]
out = {}
for k, v in sd.items():
    for pat, rep in rules:
        k2 = re.sub(pat, rep, k)
        if k2 != k:
            k = k2
            break
    out[k] = v
torch.save(out, dst)
print(f"renamed -> {dst}")
