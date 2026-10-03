"""Export official restoration checkpoints for ncnn; PyTorch is build-time only."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import textwrap
import types
from pathlib import Path

import pnnx
import torch

from photo_denoise.models import Denoiser, cache_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("drunet", "scunet", "scunet-gan"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/ncnn"))
    parser.add_argument("--fixed-size", type=int, help="Export a fixed-size graph for mask folding")
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = Denoiser(args.model, cache_directory(), "cpu", offline=True).model
    channels = 4 if args.model == "drunet" else 3
    if args.model.startswith("scunet"):
        # pnnx cannot lower Tensor.masked_fill. An additive finite attention mask
        # has identical softmax behavior here (exp(-10000) underflows to zero).
        attention = next(m for m in model.modules() if type(m).__name__ == "WMSA")
        source = textwrap.dedent(inspect.getsource(type(attention).forward))
        source = source.replace(
            'sim = sim.masked_fill_(attn_mask, float("-inf"))',
            "sim = sim + attn_mask.to(dtype=sim.dtype) * -10000.0",
        )
        namespace = dict(type(attention).forward.__globals__)
        exec(compile(source, "<ncnn-attention-export>", "exec"), namespace)
        for module in model.modules():
            if type(module).__name__ == "WMSA":
                module.forward = types.MethodType(namespace["forward"], module)
    with torch.inference_mode():
        side = args.fixed_size or 64
        pnnx.export(
            model,
            str(target / f"{args.model}.pt"),
            inputs=torch.rand(1, channels, side, side),
            inputs2=None if args.fixed_size else torch.rand(1, channels, 128, 192),
            fp16=False,
        )
    files = {suffix: target / f"{args.model}.ncnn.{suffix}" for suffix in ("param", "bin")}
    result = {}
    for suffix, path in files.items():
        with path.open("rb") as stream:
            result[suffix] = {
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": path.stat().st_size,
            }
    (target / f"{args.model}.manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
