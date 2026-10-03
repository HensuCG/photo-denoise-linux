"""Subprocess probe so optional GPU libraries cannot crash the GUI."""

import argparse
import json


def probe(runtime):
    if runtime == "vulkan":
        from .ncnn_backend import devices

        gpus = devices()
        if not gpus:
            raise ValueError("No compatible Vulkan GPU found")
        import ncnn

        return {"runtime": runtime, "gpus": gpus, "version": ncnn.__version__}
    import spandrel  # noqa: F401
    import torch

    if runtime == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; check the NVIDIA driver")
        gpus = [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count())
        ]
    else:
        gpus = []
    return {
        "runtime": runtime,
        "gpus": gpus,
        "version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=("cuda", "vulkan", "cpu"))
    args = parser.parse_args()
    print(json.dumps(probe(args.runtime)))
