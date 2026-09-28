# Arion

Arion is a small Python CLI that transcribes speech from one local audio or video file with Whisper. FFmpeg converts the input to 16 kHz mono WAV when needed.

The default output is plain text on stdout. Diagnostics go to stderr, so shell redirection and pipes work as expected. The first run downloads the selected model into the local cache.

## Install

Use Python 3.10 or newer. On Apple Silicon, MLX requires macOS 14 or newer and a native arm64 Python. Python 3.12 is a practical choice on both platforms.

### macOS (Apple Silicon)

```sh
brew install python@3.12 ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package installs `mlx-whisper` for Metal and `faster-whisper` for CPU fallback.

### Windows (NVIDIA)

Run in PowerShell after installing a compatible NVIDIA driver:

```powershell
winget install --id Python.Python.3.12 --exact
winget install --id Gyan.FFmpeg --exact
winget install --id Nvidia.CUDA --version 12.6 --exact
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[cuda]"
```

Open a new terminal after winget installs system tools. The `cuda` extra supplies CUDA runtime, cuBLAS, and cuDNN Python wheels; Arion adds their DLL directories for the process. The Windows CUDA path still needs a run on a real NVIDIA machine to confirm the version combination. Without CUDA, use `python -m pip install -e .` for CPU mode.

## Use

```sh
arion interview.mp4
arion interview.mp4 --language ru --output interview.txt
arion recording.wav --device cpu --model small > transcript.txt
```

`--language` accepts a Whisper language code such as `ru` or `en`; omitting it enables automatic language detection. The default model is `turbo`. Available models: `tiny`, `base`, `small`, `medium`, `large-v3`, and `turbo`.

`--device auto` tries Metal, then CUDA, then CPU. An explicit `--device` fails if that device is unavailable. `--output` writes a UTF-8 text file; add `--force` to replace an existing file. Use `arion --help` for the complete CLI options.

Prefer `--output transcript.txt` when you want a file: Arion creates it only after successful transcription. Shell redirection (`> transcript.txt`) creates or truncates the file before Arion starts, including when the input has no audio.

## Development

```sh
python -m unittest discover -s tests
```

The tests cover CLI behavior and FFmpeg conversion without downloading a model. Metal performance on M5 Pro and CUDA on Windows require hardware checks.

See [docs/requirements.md](docs/requirements.md) for the agreed behavior and [AGENTS.md](AGENTS.md) for development guidance.

## License

[Unlicense](LICENSE). The source is dedicated to the public domain.
