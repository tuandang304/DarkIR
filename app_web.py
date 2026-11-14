#!/usr/bin/env python3
"""
app_web.py -- Giao diện web Gradio cho DarkIR (low-light image restoration).

Chạy:
    ./venv_DarkIR/bin/python app_web.py --port 7860
    # rồi mở http://localhost:7860

Giao diện gồm 5 tab: So sánh, Ảnh, Hàng loạt, Video, Giới thiệu.
Mô hình được nạp lười (lazy) và cache lại; chỉ nạp lại khi thay đổi cài đặt.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from darkir_core import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    DarkIREnhancer,
)
from model_zoo import LIST_MODELS, get_backend, model_info


def _patch_gradio_client_bool_schema_bug() -> None:
    """Work around a gradio 5.12.0 / gradio_client mismatch.

    gradio's file-based components (Image, File, Video, DownloadButton, Gallery...)
    expose ``"additionalProperties": true`` in their API schema. The installed
    gradio_client version assumes ``additionalProperties`` is always a dict and
    raises ``TypeError: argument of type 'bool' is not iterable`` while building
    ``/info``, which makes every page request fail with HTTP 500.

    Returning "Any" for boolean schemas is correct JSON-Schema behaviour and
    leaves every normal schema untouched.
    """
    try:
        from gradio_client import utils as gc_utils

        original = getattr(gc_utils, "_json_schema_to_python_type", None)
        if original is None or getattr(original, "_darkir_schema_patch", False):
            return

        def _patched(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            return original(schema, defs)

        _patched._darkir_schema_patch = True  # type: ignore[attr-defined]
        gc_utils._json_schema_to_python_type = _patched  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - never let a compatibility patch break the UI
        pass


_patch_gradio_client_bool_schema_bug()

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = "models/DarkIR_384.pt"
_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="darkir_web_"))

# gradio's Video component shells out to ffprobe for input and output metadata.
# Without ffprobe (common in minimal CPU setups) we fall back to plain file
# upload / download so the Video tab keeps working.
HAS_FFPROBE = shutil.which("ffprobe") is not None

# --------------------------------------------------------------------------
# Model cache (lazy load, reload when settings change)
# --------------------------------------------------------------------------
_ENHANCER: DarkIREnhancer | None = None
_ENHANCER_KEY: tuple | None = None
_MODEL_LOCK = threading.Lock()

_PREVIEW_MAX_SIDE = 1100

EMPTY_PREVIEW = (
    '<div class="preview-empty">'
    "<span class=\"preview-icon\">🌙</span>"
    "<div>Ảnh so sánh trước / sau sẽ xuất hiện ở đây.</div>"
    '<div class="preview-sub">Tải lên một ảnh thiếu sáng rồi bấm “Khôi phục ảnh”.</div>'
    "</div>"
)

CMP_CHOICES = [f"{name} · {model_info(name)['paper']}" for name in LIST_MODELS()]


def _device_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = [("Tự động (auto)", "auto"), ("CPU", "cpu")]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            choices.append((f"GPU cuda:{index} · {name}", f"cuda:{index}"))
    return choices


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_device(value: str | None) -> str | None:
    if not value or value == "auto":
        return None
    return value


def get_enhancer(
    device: str | None,
    max_side: float | int | None,
    normalize: bool,
    weights: str | None,
    progress: gr.Progress | None = None,
) -> DarkIREnhancer:
    """Load (or reuse) the DarkIR model. Cache key = all inference settings."""
    global _ENHANCER, _ENHANCER_KEY

    side = int(max_side or 0)
    key = (str(weights or DEFAULT_WEIGHTS), str(device or "auto"), side, bool(normalize))
    with _MODEL_LOCK:
        if _ENHANCER is not None and _ENHANCER_KEY == key:
            return _ENHANCER
        # Release the old model before building the new one (CPU RAM friendly).
        _ENHANCER = None
        _ENHANCER_KEY = None
        if progress is not None:
            progress(0.03, desc="Đang nạp mô hình DarkIR...")
        enhancer = DarkIREnhancer(
            weights=weights or None,
            device=_resolve_device(device),
            max_side=side or None,
            normalize=bool(normalize),
        )
        _ENHANCER = enhancer
        _ENHANCER_KEY = key
        return enhancer


def invalidate_model() -> str:
    """Drop the cached model so the next run reloads with new settings."""
    global _ENHANCER, _ENHANCER_KEY
    with _MODEL_LOCK:
        _ENHANCER = None
        _ENHANCER_KEY = None
    return _stats_html(None, note="Cài đặt đã thay đổi — mô hình sẽ được nạp lại ở lần chạy tới.")


def _stats_html(enhancer: DarkIREnhancer | None = None, error: str | None = None,
                note: str | None = None) -> str:
    if error:
        dot = "err"
        device = "—"
        params = "—"
        weights = "—"
        side = "—"
        norm = "—"
        footer = f'<div class="stat-note">⚠️ {html.escape(str(error))}</div>'
    elif enhancer is not None:
        dot = "ok"
        device = str(enhancer.device)
        params = f"{enhancer.num_params / 1e6:.2f} M"
        weights = enhancer.weights_path.name
        side = f"{enhancer.max_side} px" if enhancer.max_side else "gốc"
        norm = "bật" if enhancer.normalize else "tắt"
        footer = f'<div class="stat-note">Checkpoint: <code>{html.escape(str(enhancer.weights_path))}</code></div>'
    else:
        dot = "idle"
        device = "—"
        params = "—"
        weights = "—"
        side = "—"
        norm = "—"
        footer = ""

    note_html = ""
    if note:
        note_html = f'<div class="stat-note">{html.escape(note)}</div>'

    return f"""
<div class="stat-list">
  <div class="stat-row"><span>Trạng thái</span><b><span class="dot {dot}"></span>{"Đã nạp" if enhancer is not None else ("Lỗi" if error else "Chưa nạp")}</b></div>
  <div class="stat-row"><span>Thiết bị</span><b>{html.escape(device)}</b></div>
  <div class="stat-row"><span>Tham số</span><b>{html.escape(params)}</b></div>
  <div class="stat-row"><span>Max side</span><b>{html.escape(side)}</b></div>
  <div class="stat-row"><span>Chuẩn hoá</span><b>{html.escape(norm)}</b></div>
  <div class="stat-row"><span>Checkpoint</span><b>{html.escape(weights)}</b></div>
</div>
{footer}
{note_html}
"""


def _stats_backend(backend) -> str:
    """Sidebar stats for a model_zoo backend (same layout as _stats_html)."""
    name = getattr(backend, "name", "model")
    device = str(getattr(backend, "device", "—"))
    params_value = getattr(backend, "num_params", 0) or 0
    params = f"{params_value / 1e6:.2f} M"
    ckpt = getattr(backend, "checkpoint", "") or ""
    if not ckpt:
        weights_path = getattr(backend, "weights_path", None)
        ckpt = str(weights_path) if weights_path is not None else ""
    side = getattr(backend, "max_side", None)
    side = f"{side} px" if side else "gốc"
    norm = "bật" if getattr(backend, "normalize", False) else "tắt"
    footer = (
        f'<div class="stat-note">Checkpoint: <code>{html.escape(ckpt)}</code></div>'
        if ckpt else ""
    )
    return f"""
<div class="stat-list">
  <div class="stat-row"><span>Mô hình</span><b>{html.escape(str(name))}</b></div>
  <div class="stat-row"><span>Trạng thái</span><b><span class="dot ok"></span>Đã nạp</b></div>
  <div class="stat-row"><span>Thiết bị</span><b>{html.escape(device)}</b></div>
  <div class="stat-row"><span>Tham số</span><b>{html.escape(params)}</b></div>
  <div class="stat-row"><span>Max side</span><b>{html.escape(side)}</b></div>
  <div class="stat-row"><span>Chuẩn hoá</span><b>{html.escape(norm)}</b></div>
  <div class="stat-row"><span>Checkpoint</span><b>{html.escape(str(ckpt).split("/")[-1])}</b></div>
</div>
{footer}
"""


def _model_tag(name: str) -> str:
    """File-name tag mirroring inference_cpu.py: DarkIR / RetinexFormer / HVICIDNet / ZeroDCE++."""
    return name.replace("-", "")


def _clean_selection(selected, default: str | None = None) -> list[str]:
    """Turn UI choices (optionally 'Name · paper') into known model names, in zoo order."""
    order = LIST_MODELS()
    names: list[str] = []
    for item in selected or []:
        token = item if isinstance(item, str) else str(item)
        token = token.split(" · ")[0].strip()
        if token in order and token not in names:
            names.append(token)
    if not names and default is not None:
        names = [default]
    return names


def _error_placeholder(exc: Exception) -> Path:
    """Small placeholder image shown in the gallery when a model fails."""
    width, height = 640, 320
    image = Image.new("RGB", (width, height), (12, 18, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, width - 9, height - 9], outline=(148, 163, 184), width=2)
    draw.text((24, 24), "Khong khu phuc duoc (loi mo hinh)", fill=(248, 113, 113))
    message = str(exc) or exc.__class__.__name__
    words, lines, current = message.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) > 52:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    for index, line in enumerate(lines[:7]):
        draw.text((24, 56 + index * 20), line, fill=(203, 213, 225))
    path = _TEMP_ROOT / _unique_name("compare_error", ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _coerce_path(value) -> str | None:
    """Accept str / Path / gradio file dict / file-like object."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        for key in ("path", "name", "video"):
            if value.get(key):
                return str(value[key])
    for attr in ("name", "path"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (str, Path)):
            return str(candidate)
    return str(value)


def _unique_name(prefix: str, suffix: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}{suffix}"


def _preview_data_uri(image: Image.Image, max_side: int = _PREVIEW_MAX_SIDE) -> str:
    preview = image.convert("RGB")
    if max(preview.size) > max_side:
        preview.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=84, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def comparison_html(before: Image.Image, after: Image.Image,
                    right_label: str = "DARKIR") -> str:
    """Before/after slider (pure CSS clip-path + inline JS, no extra deps)."""
    uid = uuid.uuid4().hex[:10]
    before_uri = _preview_data_uri(before)
    after_uri = _preview_data_uri(after)
    return f"""
<div class="ba-card">
  <div class="ba-wrap" id="bawrap-{uid}">
    <img class="ba-img ba-after" src="{after_uri}" alt="Ảnh khôi phục" draggable="false"/>
    <img class="ba-img ba-before" src="{before_uri}" alt="Ảnh gốc" draggable="false"
         style="clip-path: inset(0 50% 0 0);"/>
    <div class="ba-line" style="left: 50%"></div>
    <span class="ba-tag ba-tag-l">GỐC</span>
    <span class="ba-tag ba-tag-r">{html.escape(right_label)}</span>
  </div>
  <input class="ba-range" type="range" min="0" max="100" value="50"
         aria-label="Kéo để so sánh ảnh gốc và ảnh khôi phục"
         oninput="var w=document.getElementById('bawrap-{uid}');if(!w){{return;}}w.querySelector('.ba-before').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)';w.querySelector('.ba-line').style.left=this.value+'%';"/>
  <div class="ba-hint">Kéo thanh trượt để so sánh · trái: ảnh gốc · phải: {html.escape(right_label)}</div>
</div>
"""


def comparison_grid_html(original: Image.Image,
                         results: list[dict]) -> str:
    """All models side-by-side in one static grid (no slider).

    `results`: list of dicts with keys name, image (PIL | None), ok (bool),
    params (int), seconds (float | None), error (str).
    """
    cards = []
    cells: list[dict] = [{"name": "ẢNH GỐC", "sub": f"{original.width}×{original.height} px",
                          "uri": _preview_data_uri(original), "ok": True}]
    cells += results
    for cell in cells:
        name = html.escape(str(cell["name"]))
        if cell.get("ok"):
            sub = html.escape(str(cell.get("sub", "")))
            img_src = cell["uri"]
            err_html = ""
        else:
            err = html.escape(str(cell.get("error", "lỗi"))[:220])
            sub = "⚠️ lỗi"
            img_src = cell.get("uri") or ""
            err_html = f'<div class="cmp-err">{err}</div>'
        accent = ("linear-gradient(90deg,#8b5cf6,#22d3ee)"
                  if str(cell["name"]).startswith("ẢNH")
                  else "linear-gradient(90deg,#22d3ee,#8b5cf6)")
        cards.append(f"""
  <div class="cmp-card">
    <div class="cmp-img-wrap"><img class="cmp-img" src="{img_src}" alt="{name}" draggable="false"/></div>
    <div class="cmp-bar" style="background:{accent}"></div>
    <div class="cmp-name">{name}</div>
    <div class="cmp-sub">{sub}</div>
    {err_html}
  </div>""")
    body = "".join(cards)
    return f"""
<div class="cmp-grid">{body}</div>
<style>
  .cmp-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }}
  .cmp-card {{ background:rgba(15,23,42,.75); border:1px solid rgba(148,163,184,.18);
               border-radius:16px; overflow:hidden; box-shadow:0 10px 26px rgba(2,6,23,.45);
               display:flex; flex-direction:column; }}
  .cmp-img-wrap {{ background:#0b1020; min-height:240px; display:flex; align-items:center; justify-content:center; }}
  .cmp-img {{ width:100%; max-height:380px; object-fit:contain; display:block; }}
  .cmp-bar {{ height:3px; }}
  .cmp-name {{ font-weight:700; letter-spacing:.02em; padding:10px 12px 2px; font-size:1.02em; }}
  .cmp-sub {{ color:#94a3b8; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em; padding:0 12px 10px; }}
  .cmp-err {{ color:#f87171; font-size:.85em; padding:0 12px 12px; white-space:pre-wrap; }}
</style>
"""


def about_specs_html() -> str:
    weights = REPO_DIR / DEFAULT_WEIGHTS
    exists = "có sẵn" if weights.is_file() else "chưa tìm thấy"
    size = f"{weights.stat().st_size / 1e6:.1f} MB" if weights.is_file() else "—"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có (CPU only)"
    return f"""
<div class="stat-list">
  <div class="stat-row"><span>Thiết bị khả dụng</span><b>{html.escape(gpu)}</b></div>
  <div class="stat-row"><span>PyTorch</span><b>{html.escape(torch.__version__)}</b></div>
  <div class="stat-row"><span>Checkpoint</span><b>{html.escape(str(weights.relative_to(REPO_DIR)))} ({exists}, {size})</b></div>
  <div class="stat-row"><span>Kiến trúc</span><b>DarkIR-m · width 32 · 3.31 M params</b></div>
  <div class="stat-row"><span>Thư mục tạm</span><b>{html.escape(str(_TEMP_ROOT))}</b></div>
</div>
<div class="stat-note">Bảng “Thông số” trực tiếp (thiết bị / tham số / checkpoint đang dùng) nằm ở thanh bên trái và cập nhật sau mỗi lần chạy.</div>
"""


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------
def load_model_click(device, max_side, normalize, weights):
    start = time.perf_counter()
    try:
        enhancer = get_enhancer(device, max_side, normalize, weights)
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI instead of crashing
        gr.Warning(f"Không nạp được mô hình: {exc}")
        return _stats_html(None, error=str(exc))
    elapsed = time.perf_counter() - start
    return _stats_html(
        enhancer,
        note=f"Nạp mô hình trong {elapsed:.1f}s. Sẵn sàng nhận ảnh.",
    )


def run_single(image, model_name, device, max_side, normalize, weights,
               progress=gr.Progress()):
    if image is None:
        raise gr.Error("Vui lòng tải lên hoặc kéo thả một ảnh trước khi khôi phục.")
    if model_name not in LIST_MODELS():
        model_name = "DarkIR"

    pil_image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = pil_image.size
    start = time.perf_counter()

    if model_name == "DarkIR":
        try:
            model = get_enhancer(device, max_side, normalize, weights, progress=progress)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Không nạp được mô hình: {exc}") from exc
        progress(0.15, desc="Đang khôi phục ảnh...")
        try:
            restored = model.enhance_pil(pil_image)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Lỗi khi xử lý ảnh: {exc}") from exc
        stats = _stats_html(model)
    else:
        try:
            model = get_backend(model_name, device=_resolve_device(device))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Không nạp được mô hình {model_name}: {exc}") from exc
        progress(0.15, desc=f"Đang khôi phục ảnh với {model_name}...")
        try:
            restored = model.enhance_pil(pil_image, max_side=int(max_side or 0) or None)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Lỗi khi xử lý ảnh: {exc}") from exc
        stats = _stats_backend(model)
    progress(0.95, desc="Đang tạo bản xem trước...")

    output_path = _TEMP_ROOT / _unique_name("darkir_image", ".png")
    restored.save(output_path, format="PNG")

    elapsed = time.perf_counter() - start
    used_side = getattr(model, "max_side", None)
    note = ""
    if used_side and max(width, height) > used_side:
        note = f" · xử lý ở {used_side}px rồi phóng lại kích thước gốc"
    status = (
        f"**Xong trong {elapsed:.1f}s** · {model_name} · {width}×{height} → "
        f"{restored.width}×{restored.height} · {model.device}{note}"
    )

    return (
        comparison_html(pil_image, restored, right_label=model_name.upper()),
        str(output_path),
        gr.update(visible=True, value=str(output_path)),
        status,
        stats,
    )


def clear_single():
    return (
        None,
        EMPTY_PREVIEW,
        None,
        gr.update(visible=False, value=None),
        "",
    )


def run_batch(files, selected_models, device, max_side, normalize, weights,
              progress=gr.Progress()):
    if not files:
        raise gr.Error("Vui lòng chọn ít nhất một ảnh (có thể chọn nhiều ảnh cùng lúc).")

    models = _clean_selection(selected_models, default="DarkIR")

    paths: list[Path] = []
    for item in files:
        raw = _coerce_path(item)
        if not raw:
            continue
        path = Path(raw)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    if not paths:
        raise gr.Error("Không có tệp ảnh hợp lệ trong danh sách đã chọn.")

    runners: dict[str, object] = {}

    def _model_for(name: str):
        if name not in runners:
            if name == "DarkIR":
                runners[name] = get_enhancer(device, max_side, normalize, weights,
                                             progress=progress)
            else:
                runners[name] = get_backend(name, device=_resolve_device(device))
        return runners[name]

    try:
        _model_for(models[0])
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Không nạp được mô hình: {exc}") from exc

    out_dir = _TEMP_ROOT / _unique_name("batch", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    multi = len(models) > 1
    total_steps = len(paths) * len(models)
    step = 0
    start = time.perf_counter()
    gallery: list[tuple[str, str]] = []
    saved_paths: list[Path] = []
    failures: list[str] = []
    used_names: set[str] = set()

    def _unique_candidate(stem: str, suffix: str, tag: str) -> str:
        candidate = f"{stem}{tag}{suffix}"
        counter = 2
        while candidate.lower() in used_names:
            candidate = f"{stem}{tag}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate.lower())
        return candidate

    for path in paths:
        try:
            with Image.open(path) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            failures.append(f"{path.name}: {exc}")
            continue
        for model_name in models:
            step += 1
            progress(
                0.05 + 0.85 * (step - 1) / total_steps,
                desc=f"Đang xử lý {step}/{total_steps} · {path.name} · {model_name}",
            )
            try:
                model = _model_for(model_name)
                if model_name == "DarkIR":
                    restored = model.enhance_pil(original)
                else:
                    restored = model.enhance_pil(original,
                                                 max_side=int(max_side or 0) or None)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.name} · {model_name}: {exc}")
                continue
            tag = f"__{_model_tag(model_name)}" if multi else "_darkir"
            destination = out_dir / _unique_candidate(path.stem, ".png", tag)
            restored.save(destination, format="PNG")
            saved_paths.append(destination)
            caption = f"{path.name} · {model_name} · {original.width}×{original.height}"
            if multi:
                caption = f"{path.name} · {model_name} · {original.width}×{original.height}"
            else:
                caption = f"{path.name} · {original.width}×{original.height}"
            gallery.append((str(destination), caption))

    if not saved_paths:
        detail = "\n".join(failures[:5])
        raise gr.Error(f"Không xử lý được ảnh nào.\n{detail}")

    progress(0.94, desc="Đang đóng gói ZIP...")
    zip_path = _TEMP_ROOT / _unique_name("darkir_batch", ".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for file_path in saved_paths:
            archive.write(file_path, arcname=file_path.name)

    elapsed = time.perf_counter() - start
    per_image = elapsed / len(saved_paths)
    status = (
        f"**Xong {len(saved_paths)}/{total_steps} ảnh×model trong {elapsed:.1f}s** "
        f"({per_image:.1f}s/kết quả) · mô hình: {', '.join(models)} · "
        f"ZIP gồm {len(saved_paths)} ảnh"
    )
    if failures:
        status += f" · ⚠️ {len(failures)} lỗi: " + ", ".join(failures[:3])

    if "DarkIR" in runners:
        stats = _stats_html(runners["DarkIR"])
    else:
        stats = _stats_backend(runners[models[0]])

    progress(1.0, desc="Hoàn tất")
    return (
        gallery,
        gr.update(visible=True, value=str(zip_path)),
        status,
        stats,
    )


def clear_batch():
    return None, gr.update(visible=False, value=None), ""


def _finalize_video(raw_path: Path, source_path: Path) -> Path:
    """Best effort: remux/re-encode to browser-friendly H.264 and keep audio."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return raw_path
    final = raw_path.with_name(raw_path.stem + "_h264.mp4")
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(raw_path), "-i", str(source_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final),
    ]
    try:
        subprocess.run(command, check=True, timeout=3600)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        final.unlink(missing_ok=True)
        return raw_path
    raw_path.unlink(missing_ok=True)
    return final


def run_video(video, model_name, max_seconds, device, max_side, normalize, weights,
              progress=gr.Progress()):
    import cv2

    if model_name not in LIST_MODELS():
        model_name = "DarkIR"
    source_path = _coerce_path(video)
    if not source_path:
        raise gr.Error("Vui lòng tải lên một video trước khi xử lý.")
    source = Path(source_path)
    if not source.is_file():
        raise gr.Error(f"Không tìm thấy video: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise gr.Error("Không mở được video. Hãy thử định dạng MP4 (H.264).")

    try:
        src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = total_frames / fps if total_frames > 0 else None

        limit = float(max_seconds or 0)
        if limit > 0 and duration is not None and duration > limit + 1:
            gr.Warning(
                f"Video dài {duration:.0f}s — chỉ xử lý {limit:.0f}s đầu để bảo vệ CPU. "
                "Đặt giới hạn = 0 nếu muốn xử lý toàn bộ video."
            )
        if limit <= 0 and duration is not None and duration > 60:
            gr.Warning(
                f"Video dài {duration / 60:.1f} phút trên CPU có thể mất rất nhiều thời gian. "
                "Nên đặt giới hạn thời lượng."
            )

        if src_w <= 0 or src_h <= 0:
            raise gr.Error("Không đọc được kích thước video.")

        scale = 1.0
        if max_side and max(src_w, src_h) > max_side:
            scale = float(max_side) / float(max(src_w, src_h))
        out_w = max(2, int(round(src_w * scale)) // 2 * 2)
        out_h = max(2, int(round(src_h * scale)) // 2 * 2)

        try:
            if model_name == "DarkIR":
                model = get_enhancer(device, max_side, normalize, weights, progress=progress)
            else:
                model = get_backend(model_name, device=_resolve_device(device))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Không nạp được mô hình: {exc}") from exc

        progress(0.06, desc="Đang nạp mô hình...")
        frame_limit = int(round(limit * fps)) if limit > 0 else 0
        total_to_process = total_frames if total_frames > 0 else 0
        if frame_limit > 0:
            total_to_process = min(total_to_process, frame_limit) if total_to_process > 0 else frame_limit
        if total_to_process <= 0:
            total_to_process = max(1, frame_limit or 1)

        raw_path = _TEMP_ROOT / _unique_name("darkir_video", ".mp4")
        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
        )
        if not writer.isOpened():
            capture.release()
            raise gr.Error("Không tạo được tệp video đầu ra (thiếu codec mp4v?).")

        written = 0
        start = time.perf_counter()
        try:
            while True:
                if frame_limit > 0 and written >= frame_limit:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                restored = model.enhance_frame_bgr(frame)
                writer.write(restored)
                written += 1
                progress(
                    min(0.06 + 0.88 * written / total_to_process, 0.94),
                    desc=f"Đang xử lý khung hình {written}"
                         + (f"/{total_to_process}" if total_frames > 0 else ""),
                )
        finally:
            capture.release()
            writer.release()

        if written == 0:
            raw_path.unlink(missing_ok=True)
            raise gr.Error("Không đọc được khung hình nào từ video.")

        progress(0.95, desc="Đang hoàn thiện video...")
        final_path = _finalize_video(raw_path, source)
        elapsed = time.perf_counter() - start
        speed = written / elapsed if elapsed > 0 else 0
        status = (
            f"**Xong {written} khung hình trong {elapsed:.1f}s** "
            f"({speed:.2f} fps) · {model_name} · {out_w}×{out_h} · {fps:.0f} fps · {model.device}"
        )
        if final_path != raw_path:
            status += " · đã mã hoá H.264 (giữ âm thanh nếu có)"
        if scale < 1:
            status += f" · đã thu nhỏ từ {src_w}×{src_h} để chạy trên CPU"
        if frame_limit and duration is not None and duration > limit + 1:
            status += f" · chỉ xử lý {limit:.0f}s đầu"

        progress(1.0, desc="Hoàn tất")
        return (
            str(final_path),
            gr.update(visible=True, value=str(final_path)),
            status,
            _stats_html(model) if model_name == "DarkIR" else _stats_backend(model),
        )
    finally:
        capture.release()


def run_webcam(frame, model_name, device, max_side, normalize, weights):
    """Enhance one streamed webcam frame (RGB uint8 numpy)."""
    if frame is None:
        return None
    if model_name == "DarkIR":
        return get_enhancer(device, max_side, normalize, weights).enhance_array(frame)
    return get_backend(model_name, device=_resolve_device(device)).enhance_array(
        frame, max_side=max_side or None
    )


def clear_video():
    return None, None, gr.update(visible=False, value=None), ""


# --------------------------------------------------------------------------
# Compare tab (multiple models on one image)
# --------------------------------------------------------------------------
def run_compare(image, selected_models, device, max_side, normalize, export_zip,
                progress=gr.Progress()):
    if image is None:
        raise gr.Error("Vui lòng tải lên hoặc kéo thả một ảnh trước khi so sánh.")

    models = _clean_selection(selected_models)
    if not models:
        raise gr.Error("Vui lòng chọn ít nhất một mô hình để so sánh.")

    pil_image = ImageOps.exif_transpose(image).convert("RGB")
    side = int(max_side or 0) or None

    out_dir = _TEMP_ROOT / _unique_name("compare", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, float | None, Path]] = []  # (model, seconds, png)
    timings: list[tuple[str, float, int]] = []          # (model, seconds, params)
    grid_cells: list[dict] = []                          # for compare_grid_html
    errors: list[str] = []

    start_all = time.perf_counter()
    total = len(models)
    for index, name in enumerate(models):
        base = 0.05 + 0.88 * index / total
        try:
            progress(base, desc=f"Nạp mô hình {name}...")
            backend = get_backend(name, device=_resolve_device(device))
            if name == "DarkIR":
                # Normalize only exists on DarkIR; other backends ignore it.
                backend.normalize = bool(normalize)
            progress(base + 0.2 / total, desc=f"Đang khôi phục với {name}...")
            t0 = time.perf_counter()
            restored = backend.enhance_pil(pil_image, max_side=side)
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - one model failing must not kill the run
            errors.append(f"{name}: {exc}")
            results.append((name, None, _error_placeholder(exc)))
            grid_cells.append({"name": name, "ok": False, "error": f"{name}: {exc}",
                               "uri": _preview_data_uri(_error_placeholder(exc))})
            continue
        destination = out_dir / f"image__{_model_tag(name)}.png"
        restored.save(destination, format="PNG")
        results.append((name, elapsed, destination))
        params = int(getattr(backend, "num_params", 0) or 0)
        timings.append((name, elapsed, params))
        grid_cells.append({
            "name": name, "ok": True,
            "sub": f"{params / 1e6:.2f}M params · {elapsed:.2f}s",
            "uri": _preview_data_uri(restored),
        })

    progress(0.96, desc="Đang tạo bảng so sánh & ZIP...")

    params_by_model = {name: params for name, _, params in timings}

    metrics_rows = sorted(
        [
            [name, f"{params / 1e6:.2f} M", f"{elapsed:.2f} s"]
            for name, elapsed, params in timings
        ],
        key=lambda row: float(row[2].rstrip(" s")),
    )

    zip_path: Path | None = None
    if export_zip and len(timings) > 0:
        zip_path = _TEMP_ROOT / _unique_name("darkir_compare", ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for _, _, path in results:
                if path.name.startswith("image__"):
                    archive.write(path, arcname=path.name)

    elapsed_all = time.perf_counter() - start_all
    status = (
        f"**So sánh xong trong {elapsed_all:.1f}s** · "
        f"{len(timings)}/{total} mô hình OK · {pil_image.width}×{pil_image.height}"
    )
    if timings:
        fastest = min(timings, key=lambda item: item[1])
        status += f" · nhanh nhất: **{fastest[0]}** ({fastest[1]:.2f}s)"
    if errors:
        status += " · ⚠️ lỗi: " + ", ".join(e.split(":", 1)[0] for e in errors)

    side_html = comparison_grid_html(pil_image, grid_cells)

    progress(1.0, desc="Hoàn tất")
    return (
        side_html,
        metrics_rows,
        gr.update(visible=zip_path is not None, value=str(zip_path) if zip_path else None),
        status,
    )


def clear_compare():
    default_selection = [f"{n} · {model_info(n)['paper']}" for n in LIST_MODELS()]
    return (
        None,
        default_selection,
        "",
        None,
        gr.update(visible=False, value=None),
        "",
    )


# --------------------------------------------------------------------------
# Theme & CSS
# --------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
).set(
    body_background_fill="#050914",
    body_background_fill_dark="#050914",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    background_fill_primary="#0b1220",
    background_fill_secondary="#0e1729",
    block_background_fill="rgba(15, 23, 42, 0.66)",
    block_border_color="rgba(148, 163, 184, 0.16)",
    block_border_width="1px",
    block_radius="20px",
    block_shadow="0 24px 60px -34px rgba(2, 6, 23, 0.95)",
    block_label_background_fill="transparent",
    block_label_text_color="#cbd5e1",
    block_title_text_color="#e2e8f0",
    input_background_fill="rgba(2, 6, 23, 0.55)",
    input_border_color="rgba(148, 163, 184, 0.18)",
    input_border_color_focus="rgba(56, 189, 248, 0.7)",
    button_primary_background_fill="linear-gradient(135deg, #38bdf8 0%, #6366f1 55%, #a855f7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #7dd3fc 0%, #818cf8 55%, #c084fc 100%)",
    button_primary_text_color="#04121f",
    button_secondary_background_fill="rgba(30, 41, 59, 0.7)",
    button_secondary_text_color="#e2e8f0",
    border_color_primary="rgba(148, 163, 184, 0.16)",
    color_accent_soft="rgba(56, 189, 248, 0.14)",
    slider_color="#38bdf8",
    checkbox_background_color_selected="#38bdf8",
    checkbox_border_color_selected="#38bdf8",
)

CSS = """
:root { color-scheme: dark; }

html, body {
    background: #050914 !important;
}
body {
    background-image:
        radial-gradient(1200px 700px at 10% -12%, rgba(56, 189, 248, 0.16), transparent 60%),
        radial-gradient(950px 620px at 92% -6%, rgba(168, 85, 247, 0.14), transparent 58%),
        radial-gradient(900px 900px at 50% 118%, rgba(99, 102, 241, 0.10), transparent 62%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 20px 26px 54px !important;
}

/* ---------------------------------------------------------------- header */
.hero {
    position: relative;
    overflow: hidden;
    padding: 28px 30px 24px;
    border-radius: 26px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(12, 20, 38, 0.72));
    box-shadow: 0 34px 80px -42px rgba(2, 6, 23, 0.95);
    margin-bottom: 16px;
}
.hero::before {
    content: "";
    position: absolute;
    inset: -60% 45% auto -12%;
    height: 360px;
    background: radial-gradient(closest-side, rgba(56, 189, 248, 0.30), transparent);
    filter: blur(12px);
    pointer-events: none;
}
.hero-eyebrow {
    position: relative;
    letter-spacing: 0.24em;
    font-size: 10.5px;
    text-transform: uppercase;
    color: #7dd3fc;
    font-weight: 700;
}
.darkir-title {
    position: relative;
    font-size: clamp(38px, 6vw, 60px);
    line-height: 1.02;
    margin: 10px 0 6px;
    font-weight: 800;
    letter-spacing: -0.035em;
    background: linear-gradient(100deg, #7dd3fc 0%, #a5b4fc 46%, #f0abfc 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.darkir-sub {
    position: relative;
    color: #cbd5e1;
    font-size: 16.5px;
    margin: 0;
}
.hero-links {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 17px;
}
.hero-links a {
    color: #e2e8f0;
    text-decoration: none;
    font-size: 12.5px;
    line-height: 1;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(15, 23, 42, 0.62);
    transition: border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}
.hero-links a:hover {
    border-color: rgba(56, 189, 248, 0.6);
    color: #7dd3fc;
    transform: translateY(-1px);
}
.hero-pills {
    position: relative;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 15px;
}
.pill {
    font-size: 10.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.18);
    padding: 5px 10px;
    border-radius: 8px;
    background: rgba(2, 6, 23, 0.5);
}
.pill b { color: #e2e8f0; font-weight: 600; }

/* --------------------------------------------------------------- sidebar */
.sidebar {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.84), rgba(9, 14, 28, 0.84)) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 22px !important;
    padding: 14px !important;
    box-shadow: 0 24px 60px -38px rgba(2, 6, 23, 0.95) !important;
    align-self: flex-start;
}
.sidebar .block, .sidebar .form {
    box-shadow: none !important;
    background: transparent !important;
    border: none !important;
}
.sidebar h3, .sidebar h4 { letter-spacing: 0.01em; }

.sidebar .accordion {
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 14px !important;
    background: rgba(2, 6, 23, 0.30) !important;
    margin-bottom: 10px !important;
}

/* ------------------------------------------------------------ stat cards */
.stat-list { display: flex; flex-direction: column; gap: 6px; margin: 2px 0 0; }
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    font-size: 12.5px;
    padding: 7px 10px;
    border-radius: 9px;
    background: rgba(2, 6, 23, 0.48);
    border: 1px solid rgba(148, 163, 184, 0.08);
}
.stat-row span { color: #94a3b8; letter-spacing: 0.02em; white-space: nowrap; }
.stat-row b {
    color: #e2e8f0;
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    text-align: right;
    overflow-wrap: anywhere;
}
.stat-note { font-size: 11.5px; color: #94a3b8; margin-top: 8px; line-height: 1.55; }
.stat-note code { color: #7dd3fc; background: transparent; font-size: 10.5px; }
.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 99px;
    margin-right: 6px;
    vertical-align: 1px;
}
.dot.ok { background: #34d399; box-shadow: 0 0 10px #34d399; }
.dot.idle { background: #fbbf24; box-shadow: 0 0 10px #fbbf24; }
.dot.err { background: #f87171; box-shadow: 0 0 10px #f87171; }

/* ------------------------------------------------------- before / after */
.preview-empty {
    border: 1px dashed rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    padding: 46px 20px;
    text-align: center;
    color: #94a3b8;
    background: rgba(2, 6, 23, 0.35);
    font-size: 13.5px;
    line-height: 1.7;
}
.preview-icon { display: block; font-size: 34px; margin-bottom: 8px; opacity: 0.9; }
.preview-sub { color: #64748b; font-size: 12px; }

.ba-card {
    padding: 10px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 24px 60px -40px rgba(2, 6, 23, 0.95);
}
.ba-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    line-height: 0;
    background: #0b1120;
    user-select: none;
}
.ba-img {
    display: block;
    width: 100%;
    height: auto;
    user-select: none;
    -webkit-user-drag: none;
}
.ba-after { position: relative; }
.ba-before {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.ba-line {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    transform: translateX(-1px);
    background: linear-gradient(180deg, #7dd3fc, #a855f7);
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.85);
    pointer-events: none;
}
.ba-line::after {
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 34px;
    height: 34px;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    border: 2px solid rgba(248, 250, 252, 0.9);
    background: rgba(2, 6, 23, 0.55);
    box-shadow: 0 0 0 5px rgba(56, 189, 248, 0.16), 0 8px 24px rgba(2, 6, 23, 0.7);
    backdrop-filter: blur(2px);
}
.ba-tag {
    position: absolute;
    top: 12px;
    font-size: 10px;
    letter-spacing: 0.16em;
    padding: 5px 9px;
    border-radius: 7px;
    background: rgba(2, 6, 23, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.24);
    color: #e2e8f0;
    backdrop-filter: blur(4px);
    line-height: 1;
}
.ba-tag-l { left: 12px; }
.ba-tag-r { right: 12px; color: #7dd3fc; border-color: rgba(56, 189, 248, 0.45); }

.ba-range {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, #38bdf8, #a855f7);
    outline: none;
    margin: 15px 0 4px;
    cursor: ew-resize;
}
.ba-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #f8fafc;
    border: 3px solid #38bdf8;
    cursor: ew-resize;
    box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.18);
}
.ba-range::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #f8fafc;
    border: 3px solid #38bdf8;
    cursor: ew-resize;
}
.ba-hint { text-align: center; font-size: 12px; color: #94a3b8; padding: 6px 0 2px; }

/* ------------------------------------------------------------- about tab */
.about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin: 4px 0 6px;
}
.about-card {
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: linear-gradient(160deg, rgba(15, 23, 42, 0.7), rgba(9, 14, 28, 0.55));
    padding: 16px 16px 12px;
}
.about-card h3 {
    margin: 0 0 8px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #7dd3fc;
    font-weight: 700;
}
.about-card p { margin: 0 0 8px; font-size: 13px; line-height: 1.65; color: #cbd5e1; }
.about-card ul { margin: 0; padding-left: 17px; font-size: 12.5px; line-height: 1.7; color: #94a3b8; }
.about-card li b { color: #e2e8f0; font-weight: 600; }
.about-lead { font-size: 13.5px; line-height: 1.75; color: #cbd5e1; }
.warn-card h3 { color: #fbbf24; }
.warn-card p, .warn-card li { color: #cbd5e1; }

/* --------------------------------------------------------------- general */
.tab-nav {
    border-bottom: 1px solid rgba(148, 163, 184, 0.14) !important;
    margin-bottom: 14px !important;
    gap: 4px;
}
.tab-nav button {
    border-radius: 11px 11px 0 0 !important;
    font-size: 13.5px !important;
    padding: 10px 15px !important;
    color: #94a3b8 !important;
}
.tab-nav button.selected {
    color: #7dd3fc !important;
    background: rgba(56, 189, 248, 0.08) !important;
}
button.lg, .primary {
    border-radius: 12px !important;
    font-weight: 650 !important;
    letter-spacing: 0.01em;
}
button.primary {
    box-shadow: 0 14px 34px -16px rgba(56, 189, 248, 0.75) !important;
}
.gallery-item, .gallery-item img { border-radius: 12px !important; }
.status-md p { font-size: 13px; color: #cbd5e1; }
.status-md strong { color: #7dd3fc; }
.section-title h3 { margin: 2px 0 2px; font-size: 15px; }
.section-title p { color: #94a3b8; font-size: 12.5px; margin-top: 0; }
footer { opacity: 0.55; }
footer .built-with { display: none !important; }
.credit-line { text-align: center; color: #64748b; font-size: 11.5px; margin-top: 18px; }
.credit-line a { color: #94a3b8; text-decoration: none; }
.credit-line a:hover { color: #7dd3fc; }

@media (max-width: 900px) {
    .gradio-container { padding: 10px 10px 34px !important; }
    .hero { padding: 20px 18px 18px; border-radius: 20px; }
    .darkir-sub { font-size: 14.5px; }
    .hero-links a { font-size: 11.5px; }
}
"""

HEADER_HTML = """
<div class="hero">
  <div class="hero-eyebrow">CVPR 2025 · CIDAUT AI</div>
  <h1 class="darkir-title">DarkIR</h1>
  <p class="darkir-sub">Robust Low-Light Image Restoration</p>
  <div class="hero-links">
    <a href="https://github.com/Fundacion-Cidaut/DarkIR" target="_blank" rel="noopener">GitHub ↗</a>
    <a href="https://arxiv.org/abs/2412.13443" target="_blank" rel="noopener">Paper (arXiv) ↗</a>
    <a href="https://huggingface.co/Cidaut/DarkIR" target="_blank" rel="noopener">Checkpoint (HuggingFace) ↗</a>
    <a href="https://huggingface.co/spaces/Cidaut/DarkIR" target="_blank" rel="noopener">Demo gốc ↗</a>
  </div>
  <div class="hero-pills">
    <span class="pill">Thiết bị <b>CPU</b></span>
    <span class="pill">Mô hình <b>DarkIR-m</b></span>
    <span class="pill">Tham số <b>3.31 M</b></span>
    <span class="pill">All-in-one <b>sáng · nét · khử nhiễu</b></span>
  </div>
</div>
"""

ABOUT_HTML = """
<p class="about-lead">
<b>DarkIR</b> là mô hình khôi phục ảnh thiếu sáng của CIDAUT AI, công bố tại CVPR 2025.
Thay vì đi theo hướng Transformer, DarkIR dùng một CNN hiệu quả với các cơ chế attention mới
giúp mở rộng vùng tiếp nhận (receptive field), xử lý đồng thời cả ba vấn đề thường gặp trong ảnh đêm:
<b>thiếu sáng</b>, <b>nhoè do rung tay / chuyển động</b> và <b>nhiễu hạt</b>.
</p>
<div class="about-grid">
  <div class="about-card">
    <h3>Điểm chính</h3>
    <ul>
      <li><b>All-in-one:</b> một mô hình cho illumination, deblur và denoise.</li>
      <li><b>Hiệu quả:</b> DarkIR-m chỉ <b>3.31 M</b> tham số, ~7.25 GMACs.</li>
      <li><b>Attention-CNN:</b> dilated depth-wise blocks + attention đa quy mô.</li>
      <li><b>Kiến trúc U-shaped:</b> 3 mức encoder/decoder, residual connections.</li>
      <li><b>Tổng quát:</b> SOTA trên LOLBlur, LOLv2, Real-LOLBlur.</li>
    </ul>
  </div>
  <div class="about-card warn-card">
    <h3>Giới hạn cần biết</h3>
    <ul>
      <li><b>CPU rất chậm:</b> ảnh 1280px mất ~10s, 2K/4K có thể mất hàng chục giây đến vài phút.</li>
      <li><b>Ảnh lớn:</b> dùng “Max side” 960–1280px để nhanh hơn, sau đó ảnh được phóng lại kích thước gốc.</li>
      <li><b>Video:</b> xử lý từng khung hình, nên đặt giới hạn thời lượng; âm thanh chỉ được giữ nếu có ffmpeg.</li>
      <li><b>Dữ liệu huấn luyện:</b> phần lớn là ảnh tổng hợp, có thể chưa hoàn hảo với ảnh đời thực phức tạp.</li>
      <li><b>Không phải sản phẩm:</b> đây là demo nghiên cứu, không dùng cho mục đích thương mại.</li>
    </ul>
  </div>
  <div class="about-card">
    <h3>Cách dùng nhanh</h3>
    <ul>
      <li>Tab <b>Ảnh</b>: kéo thả ảnh, bấm “Khôi phục ảnh”, kéo thanh trượt để so sánh.</li>
      <li>Tab <b>Hàng loạt</b>: chọn nhiều ảnh, tải ZIP kết quả.</li>
      <li>Tab <b>Video</b>: tải video MP4, đặt giới hạn giây cho CPU.</li>
      <li>Thanh bên: đổi thiết bị, “Max side”, chuẩn hoá tương phản, checkpoint.</li>
    </ul>
  </div>
</div>
"""

CREDIT_HTML = """
<div class="credit-line">
  Ảnh / video được xử lý cục bộ trên máy của bạn · không gửi dữ liệu lên cloud.
  Mô hình &amp; mã nguồn: <a href="https://github.com/Fundacion-Cidaut/DarkIR" target="_blank" rel="noopener">Fundacion Cidaut / DarkIR</a>
  · Giao diện Gradio: DarkIR local UI.
</div>
"""

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(
    theme=THEME,
    css=CSS,
    title="DarkIR — Robust Low-Light Image Restoration",
    analytics_enabled=False,
) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # ------------------------------------------------------------ sidebar
        with gr.Column(scale=31, min_width=290, elem_classes=["sidebar"]):
            with gr.Accordion("⚙️ Cài đặt mô hình", open=True):
                device_dd = gr.Dropdown(
                    choices=_device_choices(),
                    value=_default_device(),
                    label="Thiết bị",
                    info="Tự động chọn CUDA nếu có, ngược lại dùng CPU.",
                )
                max_side_slider = gr.Slider(
                    minimum=0,
                    maximum=2560,
                    step=64,
                    value=1280,
                    label="Max side (px)",
                    info="0 = giữ nguyên kích thước gốc. 960–1280 là khuyến nghị cho CPU.",
                )
                normalize_cb = gr.Checkbox(
                    value=False,
                    label="Chuẩn hoá tương phản từng ảnh (chỉ DarkIR)",
                    info=(
                        "Min-max stretch trước khi suy luận; hữu ích với ảnh cực tối. "
                        "Lưu ý: chỉ áp dụng cho DarkIR — các mô hình khác luôn dùng "
                        "checkpoint chính thức của chúng."
                    ),
                )
                weights_tb = gr.Textbox(
                    value=DEFAULT_WEIGHTS,
                    label="Checkpoint",
                    placeholder="models/DarkIR_384.pt",
                    info="Đường dẫn tới file .pt (tương đối với thư mục repo).",
                )
                load_btn = gr.Button("Nạp / kiểm tra mô hình", variant="secondary", size="md")

            stats_html = gr.HTML(_stats_html(None))
            gr.Markdown(
                "**Mẹo** · Lần chạy đầu tiên sẽ nạp mô hình (~1–2s). "
                "Thay đổi cài đặt sẽ khiến mô hình nạp lại ở lần chạy kế tiếp. "
                "Trên CPU, hãy bắt đầu với ảnh ≤ 1280px.",
                elem_classes=["status-md"],
            )

        # ---------------------------------------------------------------- main
        with gr.Column(scale=69, min_width=520):
            with gr.Tabs():
                # ----------------------------------------------------- image
                with gr.Tab("🖼️ Ảnh / Image"):
                    with gr.Row():
                        image_in = gr.Image(
                            type="pil",
                            label="Ảnh đầu vào",
                            sources=["upload", "clipboard"],
                            height=400,
                            placeholder="Kéo thả ảnh thiếu sáng vào đây, hoặc bấm để chọn tệp…",
                            show_download_button=False,
                        )
                    with gr.Row():
                        single_model = gr.Radio(
                            choices=LIST_MODELS(),
                            value="DarkIR",
                            label="Mô hình khôi phục",
                            info=(
                                "DarkIR dùng checkpoint & tuỳ chọn chuẩn hoá ở thanh bên; "
                                "các mô hình khác dùng checkpoint chính thức của chúng "
                                "(model_zoo)."
                            ),
                        )
                    with gr.Row():
                        single_btn = gr.Button("✨ Khôi phục ảnh", variant="primary", size="lg", scale=3)
                        single_clear_btn = gr.Button("Xoá", variant="secondary", size="lg", scale=1)

                    single_status = gr.Markdown("", elem_classes=["status-md"])
                    single_preview = gr.HTML(EMPTY_PREVIEW)

                    with gr.Row():
                        single_download = gr.DownloadButton(
                            "⬇️ Tải ảnh PNG", variant="secondary", size="md", visible=False
                        )
                    with gr.Accordion("Xem ảnh kết quả đầy đủ", open=False):
                        single_result = gr.Image(
                            type="filepath",
                            label="Ảnh đã khôi phục (đầy đủ)",
                            interactive=False,
                            height=400,
                        )

                    gr.Examples(
                        examples=[
                            [str(REPO_DIR / "assets" / "qualis" / "inputs" / name)]
                            for name in ("0010.png", "0075.png", "0087.png", "0088.png")
                        ],
                        inputs=[image_in],
                        label="Ảnh mẫu",
                        examples_per_page=4,
                    )

                # ---------------------------------------------------- compare
                with gr.Tab("⚖️ So sánh / Compare"):
                    gr.Markdown(
                        "### So sánh các mô hình\n"
                        "Chạy cùng một ảnh qua nhiều mô hình khôi phục ảnh thiếu sáng "
                        "(DarkIR, RetinexFormer, HVI-CIDNet, Zero-DCE++) và so sánh kết quả, "
                        "tham số và thời gian suy luận.",
                        elem_classes=["section-title"],
                    )
                    with gr.Row(equal_height=False):
                        cmp_image = gr.Image(
                            type="pil",
                            label="Ảnh đầu vào",
                            sources=["upload", "clipboard"],
                            height=340,
                            placeholder="Kéo thả ảnh thiếu sáng vào đây…",
                        )
                        with gr.Column(scale=1, min_width=300):
                            cmp_models = gr.CheckboxGroup(
                                choices=CMP_CHOICES,
                                value=CMP_CHOICES,
                                label="Mô hình tham gia so sánh",
                                info="Bỏ chọn bớt model để chạy nhanh hơn trên CPU.",
                            )
                            cmp_export_zip = gr.Checkbox(
                                value=False,
                                label="Xuất ZIP tất cả kết quả",
                                info="Gói ảnh gốc + ảnh khôi phục của từng mô hình thành file ZIP.",
                            )
                            with gr.Row():
                                cmp_btn = gr.Button(
                                    "⟲ So sánh 4 model", variant="primary", size="lg", scale=3
                                )
                                cmp_clear_btn = gr.Button(
                                    "Xoá", variant="secondary", size="lg", scale=1
                                )
                    cmp_status = gr.Markdown("", elem_classes=["status-md"])
                    cmp_zip = gr.File(
                        label="ZIP kết quả so sánh",
                        file_types=[".zip"],
                        interactive=False,
                        visible=False,
                    )
                    cmp_grid = gr.HTML("")
                    cmp_metrics = gr.Dataframe(
                        headers=["Mô hình", "Tham số", "Thời gian"],
                        datatype=["str", "str", "str"],
                        label="Bảng so sánh (sắp xếp theo thời gian tăng dần)",
                        interactive=False,
                        wrap=True,
                    )

                # ----------------------------------------------------- batch
                with gr.Tab("🗂️ Hàng loạt / Batch"):
                    gr.Markdown(
                        "### Xử lý hàng loạt\n"
                        "Chọn nhiều ảnh cùng lúc. Kết quả hiển thị trong thư viện ảnh và "
                        "có thể tải về dưới dạng một file ZIP.",
                        elem_classes=["section-title"],
                    )
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        label="Chọn ảnh (có thể chọn nhiều)",
                        height=200,
                    )
                    batch_models = gr.CheckboxGroup(
                        choices=LIST_MODELS(),
                        value=["DarkIR"],
                        label="Mô hình",
                        info=(
                            "Chọn nhiều mô hình để chạy tất cả ảnh qua từng mô hình; "
                            "tệp trong ZIP sẽ đặt tên <ảnh>__<model>.png (giống inference_cpu.py)."
                        ),
                    )
                    with gr.Row():
                        batch_btn = gr.Button(
                            "✨ Khôi phục tất cả", variant="primary", size="lg", scale=3
                        )
                        batch_clear_btn = gr.Button("Xoá", variant="secondary", size="lg", scale=1)
                        batch_download = gr.DownloadButton(
                            "⬇️ Tải ZIP", variant="secondary", size="lg", visible=False
                        )
                    batch_status = gr.Markdown("", elem_classes=["status-md"])
                    batch_gallery = gr.Gallery(
                        label="Kết quả",
                        columns=3,
                        height="auto",
                        object_fit="contain",
                        preview=False,
                    )

                # ----------------------------------------------------- video
                with gr.Tab("🎬 Video"):
                    gr.Markdown(
                        "### Khôi phục video\n"
                        "Video được xử lý từng khung hình trên CPU. Hãy giới hạn thời lượng "
                        "để tránh chờ quá lâu. Video dài hơn giới hạn sẽ chỉ được xử lý phần đầu.",
                        elem_classes=["section-title"],
                    )
                    video_in = (
                        gr.Video(
                            label="Video đầu vào",
                            sources=["upload"],
                            height=340,
                        )
                        if HAS_FFPROBE
                        else gr.File(
                            label="Video đầu vào (MP4, MOV, AVI, MKV…)",
                            file_count="single",
                            file_types=["video"],
                            type="filepath",
                            height=160,
                        )
                    )
                    if not HAS_FFPROBE:
                        gr.Markdown(
                            "ℹ️ Môi trường này thiếu `ffprobe` nên video được chọn dưới dạng tệp "
                            "và không xem trước trực tiếp được. Video vẫn được xử lý và tải về bình thường.",
                            elem_classes=["status-md"],
                        )
                    with gr.Row():
                        video_model = gr.Radio(
                            choices=LIST_MODELS(),
                            value="DarkIR",
                            label="Mô hình khôi phục",
                            info="DarkIR dùng checkpoint ở thanh bên; mô hình khác dùng checkpoint chính thức.",
                        )
                        max_seconds_slider = gr.Slider(
                            minimum=0,
                            maximum=120,
                            step=5,
                            value=20,
                            label="Giới hạn thời lượng (giây)",
                            info="0 = xử lý toàn bộ video (rất chậm trên CPU).",
                            scale=3,
                        )
                    with gr.Row():
                        video_btn = gr.Button(
                            "✨ Khôi phục video", variant="primary", size="lg", scale=3
                        )
                        video_clear_btn = gr.Button("Xoá", variant="secondary", size="lg", scale=1)
                        video_download = gr.DownloadButton(
                            "⬇️ Tải MP4", variant="secondary", size="lg", visible=False
                        )
                    video_status = gr.Markdown("", elem_classes=["status-md"])
                    video_out = (
                        gr.Video(
                            label="Video kết quả",
                            interactive=False,
                            height=340,
                        )
                        if HAS_FFPROBE
                        else gr.File(
                            label="Video kết quả (MP4)",
                            file_count="single",
                            type="filepath",
                            interactive=False,
                            height=160,
                        )
                    )

                # ---------------------------------------------------- webcam
                with gr.Tab("📷 Webcam"):
                    gr.Markdown(
                        "### Khôi phục webcam real-time\n"
                        "Cần GPU để đạt tốc độ thời gian thực. Trình duyệt chỉ cho dùng webcam "
                        "qua HTTPS hoặc localhost (dùng `--share` khi chạy từ xa). "
                        "Giảm *Max side* (vd. 480–640) để tăng FPS.",
                        elem_classes=["section-title"],
                    )
                    webcam_model = gr.Radio(
                        choices=LIST_MODELS(), value="DarkIR", label="Mô hình khôi phục"
                    )
                    with gr.Row():
                        webcam_in = gr.Image(
                            sources=["webcam"], streaming=True, type="numpy",
                            label="Webcam", height=360,
                        )
                        webcam_out = gr.Image(label="Kết quả", interactive=False, height=360)

                # ----------------------------------------------------- about
                with gr.Tab("ℹ️ Giới thiệu / About"):
                    gr.HTML(ABOUT_HTML)
                    with gr.Accordion("📊 Thông số phiên chạy", open=True):
                        gr.HTML(about_specs_html())

    gr.HTML(CREDIT_HTML)

    # ---------------------------------------------------------------- events
    load_btn.click(
        load_model_click,
        inputs=[device_dd, max_side_slider, normalize_cb, weights_tb],
        outputs=[stats_html],
    )

    for component in (device_dd, max_side_slider, normalize_cb):
        component.change(invalidate_model, inputs=None, outputs=[stats_html])
    weights_tb.submit(invalidate_model, inputs=None, outputs=[stats_html])
    weights_tb.blur(invalidate_model, inputs=None, outputs=[stats_html])

    single_btn.click(
        run_single,
        inputs=[image_in, single_model, device_dd, max_side_slider, normalize_cb, weights_tb],
        outputs=[single_preview, single_result, single_download, single_status, stats_html],
    )
    single_clear_btn.click(
        clear_single,
        inputs=None,
        outputs=[image_in, single_preview, single_result, single_download, single_status],
    )

    batch_btn.click(
        run_batch,
        inputs=[batch_files, batch_models, device_dd, max_side_slider, normalize_cb, weights_tb],
        outputs=[batch_gallery, batch_download, batch_status, stats_html],
    )
    batch_clear_btn.click(
        clear_batch,
        inputs=None,
        outputs=[batch_files, batch_download, batch_status],
    )

    video_btn.click(
        run_video,
        inputs=[video_in, video_model, max_seconds_slider, device_dd, max_side_slider, normalize_cb, weights_tb],
        outputs=[video_out, video_download, video_status, stats_html],
    )
    video_clear_btn.click(
        clear_video,
        inputs=None,
        outputs=[video_in, video_out, video_download, video_status],
    )

    webcam_in.stream(
        run_webcam,
        inputs=[webcam_in, webcam_model, device_dd, max_side_slider, normalize_cb, weights_tb],
        outputs=[webcam_out],
        stream_every=0.1,
        time_limit=600,
        concurrency_limit=1,
        show_progress="hidden",
    )

    cmp_btn.click(
        run_compare,
        inputs=[cmp_image, cmp_models, device_dd, max_side_slider, normalize_cb, cmp_export_zip],
        outputs=[cmp_grid, cmp_metrics, cmp_zip, cmp_status],
    )
    cmp_clear_btn.click(
        clear_compare,
        inputs=None,
        outputs=[cmp_image, cmp_models, cmp_grid, cmp_metrics, cmp_zip, cmp_status],
    )

    gr.Examples(
        examples=[
            [str(REPO_DIR / "assets" / "qualis" / "inputs" / name)]
            for name in ("0010.png", "0075.png")
        ],
        inputs=[cmp_image],
        label="Ảnh mẫu cho So sánh",
        examples_per_page=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DarkIR Gradio web UI")
    parser.add_argument("--port", type=int, default=7860, help="Cổng HTTP (mặc định: 7860)")
    parser.add_argument("--host", default="0.0.0.0", help="Địa chỉ bind (mặc định: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Tạo link public qua Gradio")
    args = parser.parse_args()

    demo.queue(default_concurrency_limit=1, max_size=32)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(_TEMP_ROOT)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
