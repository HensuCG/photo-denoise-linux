"""Optional PyTorch backend; imported only for CUDA/CPU inference."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from spandrel import ImageModelDescriptor, ModelLoader

from .models import download_model
from .tiling import TiledDenoiser


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Run 'photo-denoise doctor' or choose --device cpu.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


class TorchDenoiser(TiledDenoiser):
    def __init__(self, name: str, directory: Path, device: str, *, offline: bool = False):
        self.name = name
        self.device = resolve_device(device)
        # Restrict checkpoint loading to tensors; never unpickle arbitrary model objects.
        state = torch.load(
            download_model(name, directory, offline=offline), map_location="cpu", weights_only=True
        )
        descriptor = ModelLoader().load_from_state_dict(state)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise ValueError("Expected an image restoration model")
        self.descriptor = descriptor.eval().to(self.device)
        self.model = descriptor.model

    @torch.inference_mode()
    def _predict(self, image: np.ndarray, sigma: float) -> np.ndarray:
        height, width = image.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.name == "drunet":
            # Spandrel's DRUNet descriptor hardcodes sigma=15. Call the underlying model
            # with our own noise map and the same required padding instead.
            pad_h = (-height) % 8
            pad_w = (-width) % 8
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
            noise = torch.full_like(tensor[:, :1], sigma / 255.0)
            result = self.model(torch.cat((tensor, noise), dim=1))[:, :, :height, :width]
        else:
            result = self.descriptor(tensor)
        return result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
