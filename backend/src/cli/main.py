from __future__ import annotations
import librosa
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from core.audio.dsp import (
    audio_info,
    load_audio,
    normalize_audio,
    save_spectrogram,
    save_waveform,
    to_mono,)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR.parent / "data" / "output"


def print_audio_info(metadata: dict[str, Any]) -> None:
    print(f"File: {metadata['path']}")
    print(f"Sample rate: {metadata['sample_rate']} Hz")
    print(f"Duration: {metadata['duration']:.3f} s")
    print(f"Channels: {metadata['channels']}")
    print(f"Peak amplitude: {metadata['peak']:.6f}")
    print(f"RMS amplitude: {metadata['rms']:.6f}")


def build_output_path(input_path: Path, output: Path | None, suffix: str) -> Path:
    if output is None:
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{input_path.stem}-{suffix}.png"
    if output.exists() and output.is_dir():
        return output / f"{input_path.stem}-{suffix}.png"
    return output


def main() -> None:
    parser = ArgumentParser(description="Resonance backend audio CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- existing commands ---
    info_parser = subparsers.add_parser("info", help="Print audio metadata")
    info_parser.add_argument("input_file", help="Input WAV or MP3 file path")

    waveform_parser = subparsers.add_parser("waveform", help="Generate waveform image")
    waveform_parser.add_argument("input_file", help="Input WAV or MP3 file path")
    waveform_parser.add_argument("--output", help="Output image file or directory", default=None)

    spectrogram_parser = subparsers.add_parser("spectrogram", help="Generate spectrogram image")
    spectrogram_parser.add_argument("input_file", help="Input WAV or MP3 file path")
    spectrogram_parser.add_argument("--output", help="Output image file or directory", default=None)


    ################################

    args = parser.parse_args()
    input_path = Path(args.input_file)
    metadata = load_audio(input_path)
    signal = normalize_audio(metadata["signal"])
    sr = metadata["sample_rate"]

    # info operates on the raw multi-channel signal so we handle it before to_mono
    if args.command == "info":
        mono = to_mono(signal)
        info = audio_info(mono, sr)
        info["path"] = metadata["path"]
        info["channels"] = metadata["channels"]
        print_audio_info(info)
        return

    signal = to_mono(signal)

    if args.command == "waveform":
        output_path = build_output_path(input_path, Path(args.output) if args.output else None, "waveform")
        save_waveform(signal, sr, output_path)
        print(f"Waveform saved to: {output_path}")

    elif args.command == "spectrogram":
        output_path = build_output_path(input_path, Path(args.output) if args.output else None, "spectrogram")
        save_spectrogram(signal, sr, output_path)
        print(f"Spectrogram saved to: {output_path}")




if __name__ == "__main__":
    main()