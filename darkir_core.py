"""
darkir_core.py -- Reusable, GPU-optional DarkIR inference helpers.

The official `inference.py` / `inference_video.py` scripts require NCCL + DDP
(GPU only) and expect checkpoints whose keys are prefixed with `module.`.
This module provides the same model behaviour in a plain, device-agnostic way
so it can be used from the CLI (`inference_cpu.py`) or a web UI.

Example:
    from darkir_core import DarkIREnhancer

    enhancer = DarkIREnhancer()
    out_image = enhancer.enhance_pil(pil_image)
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = REPO_DIR / "models" / "DarkIR_384.pt"

# Architecture matching the public DarkIR-m (32 width) checkpoint, which is
# also the one used by the official HuggingFace demo.
DEFAULT_ARCH = {
    "img_channel": 3,
    "width": 32,
    "middle_blk_num_enc": 2,
    "middle_blk_num_dec": 2,
    "enc_blk_nums": [1, 2, 3],
    "dec_blk_nums": [3, 1, 1],
    "dilations": [1, 4, 9],
    "extra_depth_wise": True,
}

PAD_MULTIPLE = 8
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def pick_device(device: Optional[str] = None) -> torch.device:
    """Return an explicit device, or cuda when available, else cpu."""
    if device:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


PRECISIONS = ("fp32", "fp16", "trt-fp16")


def setup_precision(model: torch.nn.Module, precision: str, device: torch.device):
    """Return (model, ctx_factory) for "fp32" | "fp16" (autocast) | "trt-fp16" (Torch-TensorRT)."""
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}. Choose from {PRECISIONS}")
    if precision == "fp32":
        return model, nullcontext
    if device.type != "cuda":
        raise ValueError(f"{precision} cần GPU CUDA (đang chọn {device}).")
    if precision == "fp16":
        return model, lambda: torch.autocast("cuda", dtype=torch.float16)
    import torch_tensorrt  # noqa: F401  registers the "torch_tensorrt" torch.compile backend

    # Engine is built on the first call and rebuilt for every new input size.
    compiled = torch.compile(
        model, backend="torch_tensorrt", dynamic=False,
        options={"enabled_precisions": {torch.float16}},
    )
    return compiled, nullcontext


def resolve_weights(weights: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve the checkpoint path, accepting None / relative paths."""
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    if not path.is_absolute():
        candidates = [Path.cwd() / path, REPO_DIR / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Download it with:\n"
            "  curl -L -o models/DarkIR_384.pt "
            "https://huggingface.co/Cidaut/DarkIR/resolve/main/DarkIR_384.pt"
        )
    return path


def load_checkpoint_state(path: Union[str, os.PathLike]) -> dict:
    """Load a checkpoint dict and return clean (unprefixed) state_dict weights."""
    checkpoints = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoints, dict):
        for key in ("params", "state_dict", "model_state_dict"):
            if key in checkpoints:
                checkpoints = checkpoints[key]
                break
    # Training checkpoints are wrapped in DDP -> strip the `module.` prefix.
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in checkpoints.items()}


class DarkIREnhancer:
    """Thin wrapper around the DarkIR network for single images / frames."""

    def __init__(
        self,
        weights: Optional[Union[str, os.PathLike]] = None,
        device: Optional[str] = None,
        width: Optional[int] = None,
        max_side: Optional[int] = None,
        normalize: bool = False,
        precision: str = "fp32",
    ) -> None:
        from archs import DarkIR

        self.device = pick_device(device)
        self.weights_path = resolve_weights(weights)
        arch = dict(DEFAULT_ARCH)
        if width:
            arch["width"] = int(width)

        self.model = DarkIR(**arch)
        state = load_checkpoint_state(self.weights_path)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.precision = precision
        self.model, self._ctx = setup_precision(self.model, precision, self.device)
        self.max_side = max_side
        self.normalize = normalize
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
        pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
        return F.pad(x, (0, pad_w, 0, pad_h), value=0)

    @staticmethod
    def _minmax(x: torch.Tensor) -> torch.Tensor:
        lo, hi = x.min(), x.max()
        if float(hi - lo) < 1e-6:
            return x
        return (x - lo) / (hi - lo)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """uint8/float HWC RGB array -> (1, 3, H, W) float tensor in [0, 1]."""
        if image.dtype == np.uint8:
            arr = image.astype(np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def enhance_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Enhance a (B, 3, H, W) batch in [0, 1]; returns same shape in [0, 1]."""
        tensor = tensor.to(self.device)
        _, _, h, w = tensor.shape
        work = self._minmax(tensor) if self.normalize else tensor

        out_hw = (h, w)
        if self.max_side and max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            out_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
            work = F.interpolate(work, size=out_hw, mode="bicubic", align_corners=False)
            h, w = out_hw

        padded = self._pad(work)
        with self._ctx():
            output = self.model(padded, side_loss=False)
        output = torch.clamp(output.float(), 0.0, 1.0)[..., :h, :w]

        if (h, w) != (tensor.shape[2], tensor.shape[3]):
            output = F.interpolate(
                output, size=(tensor.shape[2], tensor.shape[3]),
                mode="bicubic", align_corners=False,
            ).clamp(0.0, 1.0)
        return output

    # ------------------------------------------------------------- public API
    def enhance_array(self, image: np.ndarray) -> np.ndarray:
        """Enhance an RGB uint8/float HWC array and return uint8 HWC."""
        tensor = self._to_tensor(image)
        output = self.enhance_tensor(tensor)
        arr = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return (arr * 255.0).round().astype(np.uint8)

    def enhance_pil(self, image: Image.Image, max_side: Optional[int] = None) -> Image.Image:
        """Enhance a PIL image and return a PIL image."""
        if max_side is not None:
            self.max_side = int(max_side)
        return Image.fromarray(self.enhance_array(np.array(image.convert("RGB"))))

    def enhance_frame_bgr(self, frame: np.ndarray) -> np.ndarray:
        """Enhance an OpenCV BGR frame and return a BGR frame."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(self.enhance_array(rgb), cv2.COLOR_RGB2BGR)


def list_files(path: Union[str, os.PathLike], extensions) -> list[Path]:
    """List files with the given extensions (single file or recursive folder)."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
