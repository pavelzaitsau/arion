# Working on Arion

Read `docs/requirements.md` before changing behavior. Keep all project documentation in English. When changing the CLI or installation, update `README.md` and the requirements document together.

The project uses the Unlicense; see `LICENSE`. Check the terms before adding third-party source code.

## Scope

- Keep the implementation small: one local media file per invocation, with no server, batch processing, or subtitles in the first version.
- Support Apple Silicon macOS and Windows x86-64 with NVIDIA GPUs. Auto device priority is Metal → CUDA → CPU.
- Use `mlx-whisper` for Metal and `faster-whisper` for CUDA and CPU.
- Install system dependencies with Homebrew on macOS or winget on Windows, and Python dependencies with pip.

## CLI contract

- The command is `arion`. The media path is required; the language is optional.
- Send only transcript text to stdout, or to the requested file. Send progress and errors to stderr.
- Never modify the input file. Clean temporary audio on success and failure.
- Download the model on first use and reuse its cache later.
- An explicitly selected device must not silently fall back to another device.

## Verification

- Test argument parsing, conversion, output handling, and failures with small local files.
- Verify Metal on M5 Pro and CUDA on Windows using those machines. A CPU test does not establish GPU support.
- Do not commit user media, temporary WAV files, transcripts, or model caches.
