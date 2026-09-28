"""Command-line transcription with MLX Whisper or faster-whisper."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import site
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo",
}
_DLL_HANDLES = []


class ArionError(Exception):
    """A user-facing error with no traceback."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arion",
        description="Transcribe one local audio or video file with Whisper.",
        epilog="Example: arion interview.mp4 --language ru --output interview.txt",
    )
    p.add_argument("media", type=Path, metavar="MEDIA", help="audio or video file")
    p.add_argument("-l", "--language", help="Whisper language code (auto-detect if omitted)")
    p.add_argument("-o", "--output", type=Path, help="write UTF-8 text to this file")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    p.add_argument(
        "--device", choices=("auto", "metal", "cuda", "cpu"), default="auto"
    )
    p.add_argument("--model", choices=MODELS, default="turbo")
    return p


def _usable_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getframerate() == 16000
                and audio.getcomptype() == "NONE"
            )
    except (OSError, EOFError, wave.Error):
        return False


def _prepare_audio(media: Path, temporary_dir: Path) -> Path:
    if _usable_wav(media):
        return media
    output = temporary_dir / "audio.wav"
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(media), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(output),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not output.is_file():
        if "Stream map" in result.stderr and "matches no streams" in result.stderr:
            raise ArionError("input file has no audio stream")
        detail = result.stderr.strip().splitlines()
        reason = detail[-1] if detail else "conversion failed"
        raise ArionError(f"FFmpeg: {reason}")
    return output


def _metal_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        if importlib.util.find_spec("mlx_whisper") is None:
            return False
        import mlx.core as mx

        if not mx.metal.is_available():
            return False
        mx.device_info()  # Metal may be listed but unavailable to this process.
        return True
    except (ImportError, OSError, AttributeError, RuntimeError):
        return False


def _add_windows_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    roots = [Path(path) for path in site.getsitepackages()]
    roots.append(Path(site.getusersitepackages()))
    for root in roots:
        for directory in (root / "nvidia").glob("*/bin"):
            if directory.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))


def _cuda_available() -> bool:
    try:
        _add_windows_cuda_dll_dirs()
        import ctranslate2

        return (
            ctranslate2.get_cuda_device_count() > 0
            and bool(ctranslate2.get_supported_compute_types("cuda"))
        )
    except (ImportError, OSError, RuntimeError):
        return False


def _select_device(requested: str) -> str:
    if requested == "metal":
        if not _metal_available():
            raise ArionError("Metal is unavailable; install mlx-whisper on Apple Silicon")
        return "metal"
    if requested == "cuda":
        if not _cuda_available():
            raise ArionError("CUDA is unavailable; check the NVIDIA driver and CUDA libraries")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if _metal_available():
        return "metal"
    if _cuda_available():
        return "cuda"
    return "cpu"


def _transcribe(audio: Path, device: str, model: str, language: str | None) -> str:
    if device == "metal":
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio), path_or_hf_repo=MLX_MODELS[model], language=language, verbose=None
        )
        return result["text"].strip()

    try:
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as exc:
        raise ArionError("faster-whisper is unavailable; reinstall arion") from exc

    compute_type = "auto" if device == "cuda" else "int8"
    engine = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _ = engine.transcribe(str(audio), language=language, beam_size=1)
    return "".join(segment.text for segment in segments).strip()


def _validate_paths(media: Path, output: Path | None, force: bool) -> None:
    if not media.is_file():
        raise ArionError(f"input file not found: {media}")
    if output is None:
        if force:
            raise ArionError("--force requires --output")
        return
    if output.resolve() == media.resolve():
        raise ArionError("output must not overwrite the input file")
    if not output.parent.is_dir():
        raise ArionError(f"output directory not found: {output.parent}")
    if output.exists() and not force:
        raise ArionError(f"output already exists: {output} (use --force)")


def run(args: argparse.Namespace) -> None:
    _validate_paths(args.media, args.output, args.force)
    if shutil.which("ffmpeg") is None:
        raise ArionError("FFmpeg not found; install it with brew or winget")
    device = _select_device(args.device)
    print(f"Using {device}; model {args.model}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="arion-") as directory:
        audio = _prepare_audio(args.media, Path(directory))
        text = _transcribe(audio, device, args.model, args.language).strip()
    if args.output is None:
        print(text)
    else:
        mode = "w" if args.force else "x"
        with args.output.open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        print(f"arion: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
