"""
model_zoo.py -- Common interface for comparing DarkIR against other
low-light enhancement models, all running on CPU or GPU without DDP.

Models
------
- DarkIR (CVPR 2025)          : models/DarkIR_384.pt   (this repo, width=32)
- DarkIR-s (CVPR 2025)        : models/DarkIR_s.pt     (width=16, 0.87M params;
                                authors' SharePoint, not on HuggingFace)
- RetinexFormer (ICCV 2023)   : models/RetinexFormer_LOL_v2_real.pth
                                (official LOLv2-real checkpoint; architecture
                                already vendored in archs/retinexformer.py)
- Zero-DCE++ (SRP 2021)       : models/ZeroDCE++_Epoch99.pth (10.6k params)
- HVI-CIDNet (CVPR 2025)      : models/HVI_CIDNet_LOLv2_real.safetensors
                                (vendored MIT code in model_zoo/cidnet/)

Usage:
    from model_zoo import get_backend, LIST_MODELS

    backend = get_backend("HVI-CIDNet")     # loaded once, reused via cache
    out = backend.enhance_pil(img)          # single image
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from darkir_core import (
    DarkIREnhancer,
    load_checkpoint_state,
    pick_device,
    setup_precision,
    resolve_weights,
)

MODELS_DIR = REPO_DIR / "models"

# Can a model's forward tolerate any input size? RetinexFormer / DarkIR / CIDNet
# pad internally with their own multiples; Zero-DCE++ does not care.
PAD_MULTIPLE = 8


class Backend:
    """Common wrapper. Subclasses must set `name`, `paper` and implement `_forward`."""

    paper: str = ""
    checkpoint: str = ""
    num_params: int = 0
    trt_ok: bool = True  # False where TensorRT FP16 output was measured wrong on a T4

    def __init__(self, device: Optional[str] = None, precision: str = "fp32") -> None:
        if precision == "trt-fp16" and not self.trt_ok:
            raise ValueError(f"TensorRT FP16 cho {self.name} ra ảnh sai lệch — hãy chọn FP16.")
        self.device = pick_device(device)
        self.model.eval().to(self.device)
        self.precision = precision
        self.model, self._ctx = setup_precision(self.model, precision, self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self._pad = torch.nn.Identity()

    # -------------------------------------------------------------- subclasses
    def _forward(self, x: torch.Tensor) -> torch.Tensor:  # [0,1] -> [0,1]
        raise NotImplementedError

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def enhance_tensor(self, x: torch.Tensor,
                       max_side: Optional[int] = None) -> torch.Tensor:
        """Enhance (1..B, 3, H, W) in [0, 1]; returns same resolution."""
        x = x.to(self.device)
        _, _, h, w = x.shape
        work = x
        if max_side and max(h, w) > max_side:
            scale = max_side / max(h, w)
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            work = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False)
        p = (PAD_MULTIPLE - work.shape[-1] % PAD_MULTIPLE) % PAD_MULTIPLE
        q = (PAD_MULTIPLE - work.shape[-2] % PAD_MULTIPLE) % PAD_MULTIPLE
        if p or q:
            work = F.pad(work, (0, p, 0, q), value=0)
        with self._ctx():
            y = self._forward(work)
        y = torch.clamp(y.float(), 0.0, 1.0)[..., :work.shape[-2], :work.shape[-1]]
        return y[..., :h, :w]

    def enhance_array(self, image: np.ndarray,
                      max_side: Optional[int] = None) -> np.ndarray:
        """RGB uint8 HWC -> RGB uint8 HWC."""
        arr = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32) / max(image.max(), 1.0)
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
        y = self.enhance_tensor(x, max_side=max_side)
        out = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (out * 255.0).round().astype(np.uint8)

    def enhance_pil(self, image: Image.Image,
                    max_side: Optional[int] = None) -> Image.Image:
        return Image.fromarray(self.enhance_array(np.array(image.convert("RGB")), max_side=max_side))

    def enhance_frame_bgr(self, frame: np.ndarray) -> np.ndarray:
        """Enhance an OpenCV BGR frame and return a BGR frame."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(self.enhance_array(rgb), cv2.COLOR_RGB2BGR)


class DarkIRBackend(DarkIREnhancer):
    name = "DarkIR"
    paper = "CVPR 2025"
    checkpoint = "models/DarkIR_384.pt"
    width = 32

    def __init__(self, device: Optional[str] = None, precision: str = "fp32") -> None:
        super().__init__(weights=self.checkpoint, device=device, width=self.width,
                         precision=precision)
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def enhance_tensor(self, x: torch.Tensor, max_side: Optional[int] = None):
        if max_side is not None:
            self.max_side = int(max_side)
        return DarkIREnhancer.enhance_tensor(self, x)

    def enhance_array(self, image: np.ndarray, max_side: Optional[int] = None) -> np.ndarray:
        if max_side is not None:
            self.max_side = int(max_side)
        return DarkIREnhancer.enhance_array(self, image)

    def enhance_pil(self, image: Image.Image,
                    max_side: Optional[int] = None) -> Image.Image:
        """Enhance a PIL image (DarkIREnhancer-style, kept for compatibility)."""
        return Image.fromarray(self.enhance_array(np.array(image.convert("RGB")), max_side))

    def _forward(self, x):  # keep DarkIREnhancer's own padding logic
        return self.model(x, side_loss=False)


class DarkIRSmallBackend(DarkIRBackend):
    name = "DarkIR-s"
    paper = "CVPR 2025 · 0.87M"
    checkpoint = "models/DarkIR_s.pt"  # not on HF; from the authors' SharePoint models folder
    width = 16


class RetinexFormerBackend(Backend):
    name = "RetinexFormer"
    paper = "ICCV 2023 · LOLv2-real"
    checkpoint = "models/RetinexFormer_LOL_v2_real.pth"
    trt_ok = False  # mean |diff| 4.6/255 vs FP32

    def __init__(self, device: Optional[str] = None, precision: str = "fp32") -> None:
        from archs.retinexformer import RetinexFormer

        self.device = pick_device(device)
        self.model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40,
                                   stage=1, num_blocks=[1, 2, 2])
        weights = load_checkpoint_state(resolve_weights(self.checkpoint))
        self.model.load_state_dict(weights, strict=True)
        super().__init__(device, precision)
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def _forward(self, x):
        return self.model(x)  # model pads itself to /4


class ZerodcePPBackend(Backend):
    name = "Zero-DCE++"
    paper = "SRP 2021 · DCE++"
    checkpoint = "models/ZeroDCE++_Epoch99.pth"

    def __init__(self, device: Optional[str] = None, precision: str = "fp32") -> None:
        from model_zoo.zerodcepp import ZerodcePP

        self.device = pick_device(device)
        self.model = ZerodcePP(scale_factor=12)
        self.model.load_state_dict(
            torch.load(resolve_weights(self.checkpoint), map_location="cpu", weights_only=False),
            strict=True)
        super().__init__(device, precision)
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def _forward(self, x):
        return self.model(x)


class CIDNetBackend(Backend):
    name = "HVI-CIDNet"
    paper = "CVPR 2025 · LOLv2-real"
    checkpoint = "models/HVI_CIDNet_LOLv2_real.safetensors"
    trt_ok = False  # mean |diff| 36.6/255 vs FP32

    def __init__(self, device: Optional[str] = None, precision: str = "fp32") -> None:
        from model_zoo.cidnet.CIDNet import CIDNet

        self.device = pick_device(device)
        self.model = CIDNet()
        from safetensors.torch import load_file
        path = resolve_weights(self.checkpoint)
        self.model.load_state_dict(load_file(str(path)), strict=True)
        super().__init__(device, precision)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        # Following the official eval_hf.py defaults
        self.alpha_s = 1.0
        self.alpha_i = 1.0
        self.gamma = 1.0
        self.gated = True
        self.gated2 = True

    def _forward(self, x):
        trans = self.model.trans
        old = (getattr(trans, "alpha_s", self.alpha_s), getattr(trans, "alpha", self.alpha_i))
        trans.alpha_s = self.alpha_s
        trans.alpha = self.alpha_i
        try:
            y = self.model(x ** self.gamma)
        finally:
            trans.alpha_s, trans.alpha = old
        return y


_ORDER = ["DarkIR", "DarkIR-s", "RetinexFormer", "HVI-CIDNet", "Zero-DCE++"]
_CLASSES = {
    "DarkIR": DarkIRBackend,
    "DarkIR-s": DarkIRSmallBackend,
    "RetinexFormer": RetinexFormerBackend,
    "HVI-CIDNet": CIDNetBackend,
    "Zero-DCE++": ZerodcePPBackend,
}
# Optional friendly metadata used by the UI/CLI
_META = {
    "DarkIR": {"paper": "CVPR 2025", "ckpt": "DarkIR_384.pt"},
    "DarkIR-s": {"paper": "CVPR 2025 (0.87M)", "ckpt": "DarkIR_s.pt"},
    "RetinexFormer": {"paper": "ICCV 2023", "ckpt": "RetinexFormer_LOL_v2_real.pth"},
    "HVI-CIDNet": {"paper": "CVPR 2025", "ckpt": "HVI_CIDNet_LOLv2_real.safetensors"},
    "Zero-DCE++": {"paper": "SRP 2021 (DCE++)", "ckpt": "ZeroDCE++_Epoch99.pth"},
}


def LIST_MODELS() -> list[str]:
    return list(_ORDER)


def model_info(name: str) -> dict:
    return {"name": name, **_META.get(name, {})}


def get_backend(name: str, device: Optional[str] = None, precision: str = "fp32") -> Backend:
    """Instantiate (or reuse a cached instance of) a backend."""
    if name not in _CLASSES:
        raise ValueError(f"Unknown model '{name}'. Available: {_ORDER}")
    cache = _agent_cache()
    key = (name, str(device), precision)
    if key in cache:
        return cache[key]
    backend = _CLASSES[name](device=device, precision=precision)
    cache[key] = backend
    return backend


def _agent_cache() -> dict:
    global _BACKEND_CACHE
    try:
        return _BACKEND_CACHE
    except NameError:
        _BACKEND_CACHE = {}
        return _BACKEND_CACHE
