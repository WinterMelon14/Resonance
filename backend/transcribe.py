"""
python -m backend.transcribe transcribe path/to/audio.wav
python -m backend.transcribe evaluate-midi path/to/truth.mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from backend.config import CFG, ExperimentConfig
from backend.decoder import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_INFERENCE_BATCH_SIZE,
    PianoDecoder,
    evaluate_midi_file,
    transcribe_audio_to_midi,
)
from backend.models.model import PianoTranscriber
import os
import pathlib


def load_checkpoint_cross_platform(
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    if os.name != "nt":
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    finally:
        pathlib.PosixPath = original_posix_path


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _nested_value(mapping: dict, *keys: str):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value

# Reject inference settings that can silently reinterpret trained weights
def _validate_checkpoint_config(checkpoint: dict, cfg: ExperimentConfig) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        return

    comparisons = {
        "model.d_model": (
            _nested_value(saved, "model", "d_model"),
            cfg.model.d_model,
        ),
        "model.n_heads": (
            _nested_value(saved, "model", "n_heads"),
            cfg.model.n_heads,
        ),
        "feature.cqt_bins_per_octave": (
            _nested_value(saved, "feature", "cqt_bins_per_octave"),
            cfg.feature.cqt_bins_per_octave,
        ),
        "feature.midi_low": (
            _nested_value(saved, "feature", "midi_low"),
            cfg.feature.midi_low,
        ),
        "feature.midi_high": (
            _nested_value(saved, "feature", "midi_high"),
            cfg.feature.midi_high,
        ),
    }
    mismatches = [
        f"{name}: checkpoint={saved_value!r}, current={current_value!r}"
        for name, (saved_value, current_value) in comparisons.items()
        if saved_value is not None and saved_value != current_value
    ]
    if mismatches:
        raise ValueError(
            "Current CFG is incompatible with the checkpoint:\n  "
            + "\n  ".join(mismatches)
        )


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    cfg: ExperimentConfig = CFG,
) -> PianoTranscriber:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if cfg.feature.kind != "cqt":
        raise ValueError("The trained harmonic baseline requires CQT features.")

    checkpoint = load_checkpoint_cross_platform(
        checkpoint_path,
        device,
    )
    
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}")
    _validate_checkpoint_config(checkpoint, cfg)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = PianoTranscriber(cfg=cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class PianoTranscriptionPipeline:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path = CFG.paths.checkpoint_path,
        threshold_results_path: str | Path | None = None,
        device: str = "auto",
        cfg: ExperimentConfig = CFG,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold_results_path = (
            self.checkpoint_path.parent / "threshold_results.csv"
            if threshold_results_path is None
            else Path(threshold_results_path)
        )
        self.decoder = PianoDecoder(cfg, self.threshold_results_path)
        self.model = load_model(self.checkpoint_path, self.device, cfg)

    def transcribe_wav(
        self,
        wav_path: str | Path,
        output_midi_path: str | Path | None = None,
        *,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        hop_frames: int | None = None,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        blend_mode: str = "triangular",
    ) -> Path:
        wav_path = Path(wav_path)
        output_midi_path = (
            Path(self.cfg.paths.work_dir) / f"{wav_path.stem}.mid"
            if output_midi_path is None
            else Path(output_midi_path)
        )
        return transcribe_audio_to_midi(
            self.model,
            wav_path,
            output_midi_path,
            decoder=self.decoder,
            cfg=self.cfg,
            chunk_frames=chunk_frames,
            hop_frames=hop_frames,
            inference_batch_size=inference_batch_size,
            blend_mode=blend_mode,
            device=self.device,
        )

    def evaluate_midi(
        self,
        midi_path: str | Path,
        *,
        rendered_wav_path: str | Path | None = None,
        predicted_midi_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
        matrices_path: str | Path | None = None,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        hop_frames: int | None = None,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        blend_mode: str = "triangular",
    ) -> dict:
        midi_path = Path(midi_path)
        work_dir = Path(self.cfg.paths.work_dir)
        rendered_wav_path = (
            work_dir / f"{midi_path.stem}_rendered.wav"
            if rendered_wav_path is None
            else Path(rendered_wav_path)
        )
        predicted_midi_path = (
            work_dir / f"{midi_path.stem}_predicted.mid"
            if predicted_midi_path is None
            else Path(predicted_midi_path)
        )
        metrics_path = (
            work_dir / f"{midi_path.stem}_metrics.json"
            if metrics_path is None
            else Path(metrics_path)
        )
        matrices_path = (
            work_dir / f"{midi_path.stem}_matrices.npz"
            if matrices_path is None
            else Path(matrices_path)
        )
        return evaluate_midi_file(
            self.model,
            midi_path,
            rendered_wav_path=rendered_wav_path,
            predicted_midi_path=predicted_midi_path,
            metrics_path=metrics_path,
            matrices_path=matrices_path,
            decoder=self.decoder,
            cfg=self.cfg,
            chunk_frames=chunk_frames,
            hop_frames=hop_frames,
            inference_batch_size=inference_batch_size,
            blend_mode=blend_mode,
            device=self.device,
            concert_pitch_hz=440.0
        )


def _add_common_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(CFG.paths.checkpoint_path),
        help="Model checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold-results",
        type=Path,
        default=None,
        help="Threshold CSV; defaults to threshold_results.csv beside the checkpoint",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or mps (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="Sliding inference chunk length (default: %(default)s)",
    )
    parser.add_argument(
        "--hop-frames",
        type=int,
        default=None,
        help="Sliding hop; defaults to half of --chunk-frames",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help="Inference chunks per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--blend-mode",
        choices=("uniform", "triangular"),
        default="triangular",
        help="Overlap blending mode (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe piano WAV files and evaluate MIDI round trips."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe a WAV file to MIDI",
    )
    transcribe_parser.add_argument("wav", type=Path, help="Input WAV file")
    transcribe_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MIDI; defaults to CFG.paths.work_dir/<wav-stem>.mid",
    )
    _add_common_inference_arguments(transcribe_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate-midi",
        help="Render MIDI to WAV, transcribe it, and evaluate against the MIDI truth",
    )
    evaluate_parser.add_argument("midi", type=Path, help="Ground-truth MIDI file")
    evaluate_parser.add_argument("--rendered-wav", type=Path, default=None)
    evaluate_parser.add_argument("--predicted-midi", type=Path, default=None)
    evaluate_parser.add_argument("--metrics-json", type=Path, default=None)
    evaluate_parser.add_argument("--matrices-npz", type=Path, default=None)
    _add_common_inference_arguments(evaluate_parser)
    return parser


def _make_pipeline(args: argparse.Namespace) -> PianoTranscriptionPipeline:
    return PianoTranscriptionPipeline(
        checkpoint_path=args.checkpoint,
        threshold_results_path=args.threshold_results,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = _make_pipeline(args)
    inference_options = {
        "chunk_frames": args.chunk_frames,
        "hop_frames": args.hop_frames,
        "inference_batch_size": args.batch_size,
        "blend_mode": args.blend_mode,
    }

    if args.command == "transcribe":
        output_path = pipeline.transcribe_wav(
            args.wav,
            args.output,
            **inference_options,
        )
        print(
            json.dumps(
                {
                    "input_wav": str(args.wav),
                    "output_midi": str(output_path),
                    "selected_thresholds": pipeline.decoder.selected_thresholds,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-midi":
        result = pipeline.evaluate_midi(
            args.midi,
            rendered_wav_path=args.rendered_wav,
            predicted_midi_path=args.predicted_midi,
            metrics_path=args.metrics_json,
            matrices_path=args.matrices_npz,
            **inference_options,
        )
        print(json.dumps(result, indent=2))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())