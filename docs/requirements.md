# Arion requirements

Status: agreed product behavior. Updated: 2026-09-28.

## Goal

A small Python CLI transcribes speech from one local audio or video file into text in the original language. The language is optional and detected automatically when omitted. Translation, subtitles, and directory processing are outside the first version.

## CLI

```text
arion MEDIA [--language LANG] [--output PATH] [--force] [--device auto|metal|cuda|cpu] [--model MODEL]
```

- `MEDIA` is an existing local file.
- `--language`, `-l` takes a Whisper language code such as `ru` or `en`.
- By default, print the final text and a newline to stdout. Send diagnostics to stderr so redirection and pipes work.
- `--output`, `-o` writes UTF-8 text to a file instead of stdout. Its parent directory must exist. Refuse to overwrite an existing file unless `--force` is set. Never permit the output to be the input file.
- `--device` defaults to `auto`. An explicit device fails if unavailable.
- `--model` defaults to `turbo`. Supported names are `tiny`, `base`, `small`, `medium`, `large-v3`, and `turbo`, with corresponding models on both backends.
- `--help` includes the options and an example.

## Media handling

1. Validate the input file and FFmpeg availability.
2. Pass through a WAV file that is already mono, 16-bit PCM, and 16 kHz. Otherwise use FFmpeg to extract the first audio stream into a temporary WAV with those properties and no video.
3. Transcribe it and return text without timestamps or logs in stdout.
4. Remove temporary audio on both success and failure. Never change the source file.

The Python standard library can inspect WAV headers, so `ffprobe` is not required. Supported input formats are those FFmpeg can decode. A file with no audio stream must produce a clear error.

## Acceleration and models

In `auto` mode the priority is Metal → CUDA → CPU. Select an accelerator only if it is available to the running process. Fall back when hardware or its backend is unavailable; do not retry on CPU after recognition has already started and failed.

- Metal: `mlx-whisper` on Apple Silicon macOS, targeting M5 Pro.
- CUDA: `faster-whisper` on Windows with an NVIDIA GPU.
- CPU: `faster-whisper` on both systems.

The multilingual Whisper `turbo` model is the default speed and quality tradeoff. The first invocation may download several gigabytes and needs internet access; later runs use the local cache. The transcript stays in the spoken language.

## Installation

- Python 3.10 or newer; MLX needs macOS 14 or newer and native arm64 Python.
- First-version platforms: Apple Silicon macOS and Windows x86-64 with NVIDIA. CPU mode works on both.
- Install FFmpeg with Homebrew (`brew install ffmpeg`) on macOS or winget (`winget install --id Gyan.FFmpeg --exact`) on Windows.
- Install Python packages with pip. macOS uses both `mlx-whisper` and `faster-whisper`; Windows uses `faster-whisper`.
- Windows CUDA needs a compatible NVIDIA driver, CUDA 12 runtime, cuBLAS, and cuDNN. The setup must use winget and/or pip without requiring manual DLL copying. Verify the exact version combination on Windows hardware.
- No Docker or source builds are required.

## Acceptance criteria

- Invalid paths, absent audio, FFmpeg failures, unsupported language or model, unavailable explicitly selected device, and transcription failures produce a nonzero exit code and a concise stderr message.
- On Apple Silicon, `auto` uses Metal; on a working Windows NVIDIA setup, it uses CUDA; otherwise it uses CPU.
- For a readable media file, the CLI writes transcript text to stdout or the requested file and leaves no temporary audio.
- Stdout contains no diagnostics.

## Sources for the technical choices

- [MLX for Apple Silicon](https://github.com/ml-explore/mlx)
- [MLX Whisper Python API](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md)
- [faster-whisper and CUDA requirements](https://github.com/SYSTRAN/faster-whisper)
- [Whisper turbo model](https://github.com/openai/whisper/blob/main/README.md)
- [WinGet FFmpeg package](https://github.com/microsoft/winget-pkgs/blob/master/manifests/g/Gyan/FFmpeg/5.0/Gyan.FFmpeg.yaml)

## Hardware verification still needed

Run a short real transcription on M5 Pro and Windows with NVIDIA. In particular, verify that CUDA DLLs installed through winget/pip are visible to the Windows process. Software support alone does not establish speed on a specific computer.
