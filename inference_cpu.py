#!/usr/bin/env python3
"""
inference_cpu.py -- DarkIR inference without DDP/NCCL (works on CPU and GPU).

Unlike the official `inference.py` / `inference_video.py`, this script does not
use `torch.distributed`, so it runs on machines without CUDA.

Examples:
    # one image or a whole folder (recursive)
    python inference_cpu.py -i ./images/inputs
    python inference_cpu.py -i ./photo.jpg -o ./out

    # a video (audio is kept when ffmpeg is available)
    python inference_cpu.py -i ./videos/inputs/clip.mp4
    python inference_cpu.py -i ./clip.mp4 --side-by-side

    # limit the working resolution for very large inputs (much faster)
    python inference_cpu.py -i ./images/inputs --max-side 1280
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from darkir_core import (
    DEFAULT_WEIGHTS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    DarkIREnhancer,
    list_files,
)
from model_zoo import LIST_MODELS, get_backend

REPO_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DarkIR low-light restoration for images and videos (CPU or GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Image, folder of images, or video file")
    parser.add_argument("-o", "--output", default=None, help="Output folder (default: images/results or videos/results)")
    parser.add_argument("-w", "--weights", default=str(DEFAULT_WEIGHTS), help="Path to the DarkIR checkpoint")
    parser.add_argument("--width", type=int, default=None, help="Model width (32 for DarkIR-m, 64 for DarkIR-l)")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps ... (default: auto)")
    parser.add_argument("--model", action="append", choices=LIST_MODELS(),
                        help="Model to use; repeat the flag to select several. Default: DarkIR")
    parser.add_argument("--compare", action="store_true",
                        help="Shorthand for running all 4 models and comparing outputs side by side")
    parser.add_argument("--max-side", type=int, default=None,
                        help="Downscale the longest side to this value before inference (0/None = keep original)")
    parser.add_argument("--normalize", action="store_true",
                        help="Per-image min-max contrast stretch before inference (matches the official video script)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Video only: write original and restored frames side by side")
    parser.add_argument("--suffix", default="", help="Suffix added to output file names")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser


def prepare_output_dir(input_path: Path, output: str | None, is_video: bool) -> Path:
    if output:
        out_dir = Path(output).expanduser().resolve()
    elif is_video:
        out_dir = REPO_DIR / "videos" / "results"
    else:
        out_dir = REPO_DIR / "images" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def process_images(backends: list, files: list[Path], input_root: Path,
                   out_dir: Path, args) -> list[Path]:
    from tqdm import tqdm

    multi = len(backends) > 1
    results: list[Path] = []
    start = time.perf_counter()
    for path in tqdm(files, desc="Images", unit="img"):
        rel = path.relative_to(input_root) if input_root.is_dir() else Path(path.name)
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            for backend in backends:
                tag = f"__{backend.name.replace('-', '')}" if multi else args.suffix
                target = out_dir / rel.with_name(f"{rel.stem}{tag}{rel.suffix}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if multi and target.exists():
                    continue  # comparison loop overwrites silently
                if target.exists() and not args.overwrite:
                    tqdm.write(f"skip (exists): {target}")
                    continue
                restored = backend.enhance_pil(rgb, max_side=args.max_side)
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    restored.save(target, quality=95, subsampling=0)
                else:
                    restored.save(target)
                results.append(target)
    elapsed = time.perf_counter() - start
    if files:
        print(f"\n{len(results)} output(s) from {len(files)} image(s) x {len(backends)} model(s) "
              f"in {elapsed:.1f}s -> {out_dir}")
    return results


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def mux_audio(video_without_audio: Path, source_video: Path) -> bool:
    """Copy the audio track of the source video into the result (best effort)."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        return False
    tmp = video_without_audio.with_name(video_without_audio.stem + "_mux.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video_without_audio), "-i", str(source_video),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(video_without_audio)
    return True


def process_video(enhancer: DarkIREnhancer, path: Path, out_dir: Path, args) -> Path:
    import cv2
    from tqdm import tqdm

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target = out_dir / f"{path.stem}{args.suffix}.mp4"
    if target.exists() and not args.overwrite:
        print(f"skip (exists): {target}")
        cap.release()
        return target

    tmp = Path(tempfile.mkstemp(suffix=".mp4", dir=str(out_dir))[1])
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width * 2 if args.side_by_side else width, height),
    )

    start = time.perf_counter()
    written = 0
    try:
        with tqdm(total=total, desc=path.name, unit="frame") as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                restored = enhancer.enhance_frame_bgr(frame)
                writer.write(np.hstack((frame, restored)) if args.side_by_side else restored)
                written += 1
                bar.update(1)
    finally:
        cap.release()
        writer.release()

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"No frames decoded from {path}")

    tmp.replace(target)
    if args.side_by_side:
        target = _add_audio_and_maybe_reencode(target, path, fps, width, height)
    else:
        mux_audio(target, path)

    elapsed = time.perf_counter() - start
    print(f"\n{written} frames in {elapsed:.1f}s ({written / elapsed:.2f} fps) -> {target}")
    return target


def _add_audio_and_maybe_reencode(target: Path, source: Path, fps: float,
                                  width: int, height: int) -> Path:
    """Side-by-side output: re-encode with ffmpeg + source audio when possible."""
    if not _ffmpeg_bin():
        return target
    final = target.with_name(target.stem + "_sbs.mp4")
    cmd = [
        _ffmpeg_bin(), "-y", "-loglevel", "error",
        "-i", str(target), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError):
        final.unlink(missing_ok=True)
        return target
    target.unlink(missing_ok=True)
    return final


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    model_names = LIST_MODELS() if args.compare else (args.model or ["NightClear"])

    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        is_video = True
        files = [input_path]
        if len(model_names) > 1:
            print("warning: --compare for video is limited to one model per run; "
                  f"using '{model_names[0]}'", file=sys.stderr)
            model_names = [model_names[0]]
    elif input_path.is_dir():
        images = list_files(input_path, IMAGE_EXTENSIONS)
        videos = list_files(input_path, VIDEO_EXTENSIONS)
        if images and videos:
            print("error: folder mixes images and videos, run them separately", file=sys.stderr)
            return 2
        files, is_video = (videos, True) if videos else (images, False)
        if not files:
            print(f"error: no supported files in {input_path}", file=sys.stderr)
            return 2
    else:
        is_video = False
        files = [input_path]

    out_dir = prepare_output_dir(input_path, args.output, is_video)

    print(f"Loading model(s): {', '.join(model_names)}")
    backends = [get_backend(name, device=args.device) for name in model_names]
    for backend in backends:
        print(f"  - {backend.name:15s} {backend.paper:22s} {backend.num_params / 1e6:6.2f} M params")
    if args.max_side:
        print(f"Max side: {args.max_side}px (internal downscale/upscale)")

    if is_video:
        for video in files:
            process_video(backends[0], video, out_dir, args)
    else:
        process_images(backends, files, input_path, out_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
