import io
import shutil
import subprocess
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from arion import cli


def make_wav(path: Path, *, channels: int = 1, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * (channels * rate // 10))


class CliTests(unittest.TestCase):
    def test_auto_device_priority(self):
        with patch.object(cli, "_metal_available", return_value=True), patch.object(
            cli, "_cuda_available", return_value=True
        ):
            self.assertEqual(cli._select_device("auto"), "metal")
        with patch.object(cli, "_metal_available", return_value=False), patch.object(
            cli, "_cuda_available", return_value=True
        ):
            self.assertEqual(cli._select_device("auto"), "cuda")
        with patch.object(cli, "_metal_available", return_value=False), patch.object(
            cli, "_cuda_available", return_value=False
        ):
            self.assertEqual(cli._select_device("auto"), "cpu")

    def test_matching_wav_skips_conversion_and_prints_only_text(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "speech.wav"
            make_wav(media)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(cli, "_select_device", return_value="cpu"), patch.object(
                cli, "_transcribe", return_value=" hello "
            ) as transcribe, patch.object(cli.subprocess, "run") as ffmpeg, redirect_stdout(
                stdout
            ), redirect_stderr(stderr):
                status = cli.main([str(media)])
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), "hello\n")
            self.assertIn("Using cpu", stderr.getvalue())
            self.assertEqual(transcribe.call_args.args[0], media)
            ffmpeg.assert_not_called()

    def test_conversion_and_temporary_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "speech.wav"
            make_wav(media, channels=2, rate=44100)
            output = Path(directory) / "speech.txt"
            prepared = []

            def fake_transcribe(path, *_):
                with wave.open(str(path), "rb") as audio:
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertEqual(audio.getframerate(), 16000)
                prepared.append(path)
                return "recognized"

            with patch.object(cli, "_select_device", return_value="cpu"), patch.object(
                cli, "_transcribe", side_effect=fake_transcribe
            ), redirect_stderr(io.StringIO()):
                status = cli.main([str(media), "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "recognized\n")
            self.assertFalse(prepared[0].exists())
            self.assertTrue(media.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "speech.wav"
            output = Path(directory) / "speech.txt"
            make_wav(media)
            output.write_text("old", encoding="utf-8")
            stderr = io.StringIO()
            with patch.object(cli, "_transcribe") as transcribe, redirect_stderr(stderr):
                status = cli.main([str(media), "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "old")
            self.assertIn("already exists", stderr.getvalue())
            transcribe.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_video_without_audio_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "silent.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=black:s=16x16:d=1", "-an", "-y", str(media),
                ],
                check=True,
            )
            stderr = io.StringIO()
            with patch.object(cli, "_select_device", return_value="cpu"), redirect_stderr(
                stderr
            ):
                status = cli.main([str(media)])
            self.assertEqual(status, 1)
            self.assertIn("no audio stream", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
