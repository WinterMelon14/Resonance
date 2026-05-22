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
    to_mono,notes_to_midi,
)
from core.audio.dsp import (
    compute_fft_piano,
    detect_onsets_piano,
    estimate_note_pitches,
    save_fft_plot,
    save_onset_plot,
)

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

    # --- new commands ---
    fft_parser = subparsers.add_parser("fft", help="Generate FFT frequency spectrum (piano range)")
    fft_parser.add_argument("input_file", help="Input WAV or MP3 file path")
    fft_parser.add_argument("--output", help="Output image file or directory", default=None)
    fft_parser.add_argument("--fmin", type=float, default=27.5, help="Min frequency in Hz (default: 27.5 = A0)")
    fft_parser.add_argument("--fmax", type=float, default=4186.0, help="Max frequency in Hz (default: 4186 = C8)")

    onsets_parser = subparsers.add_parser("onsets", help="Detect note onsets and save annotated waveform")
    onsets_parser.add_argument("input_file", help="Input WAV or MP3 file path")
    onsets_parser.add_argument("--output", help="Output image file or directory", default=None)
    onsets_parser.add_argument("--delta", type=float, default=0.07, help="Onset sensitivity threshold (default: 0.07)")
    onsets_parser.add_argument("--hop-length", type=int, default=256, dest="hop_length", help="Hop length in samples (default: 256)")

    notes_parser = subparsers.add_parser("notes", help="Estimate MIDI note at each onset")
    notes_parser.add_argument("input_file", help="Input WAV or MP3 file path")
    notes_parser.add_argument("--output", help="Save results as JSON file", default=None)
    notes_parser.add_argument("--frame-duration", type=float, default=0.08, dest="frame_duration", help="Seconds to analyse after each onset (default: 0.08)")
    notes_parser.add_argument("--delta", type=float, default=0.07, help="Onset sensitivity passed to detector (default: 0.07)")

    # --- build notes --- 
    midi_parser = subparsers.add_parser("midi", help="Convert detected notes to a MIDI file")
    midi_parser.add_argument("input_file", help="Input WAV or MP3 file path")
    midi_parser.add_argument("--output", help="Output .mid file path", default=None)
    midi_parser.add_argument("--tempo", type=int, default=120, help="BPM for the MIDI file (default: 120)")
    midi_parser.add_argument("--delta", type=float, default=0.07, help="Onset sensitivity (default: 0.07)")
    midi_parser.add_argument("--frame-duration", type=float, default=0.08, dest="frame_duration")

    # play midi
    play_parser = subparsers.add_parser("play", help="Play a MIDI file from the output directory")
    play_parser.add_argument("midi_file", help="MIDI filename or full path")

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

    elif args.command == "fft":
        output_path = build_output_path(input_path, Path(args.output) if args.output else None, "fft")
        fft_data = compute_fft_piano(signal, sr, freq_range=(args.fmin, args.fmax))
        print(output_path)
        save_fft_plot(fft_data, output_path)
        peak_freq = fft_data["frequencies"][fft_data["magnitudes"].argmax()]
        print(f"FFT plot saved to:  {output_path}")
        print(f"Dominant frequency: {peak_freq:.1f} Hz  ({_hz_to_note(peak_freq)})")

    elif args.command == "onsets":
        output_path = build_output_path(input_path, Path(args.output) if args.output else None, "onsets")
        harmonic, percussive = librosa.effects.hpss(signal)

        onset_times = detect_onsets_piano(percussive, sr, delta=args.delta, hop_length=args.hop_length)
        save_onset_plot(signal, sr, onset_times, output_path)
        print(f"Onset plot saved to: {output_path}")
        print(f"Onsets detected:     {len(onset_times)}")
        print("  " + "  ".join(f"{t:.3f}s" for t in onset_times[:16])
              + ("  …" if len(onset_times) > 16 else ""))

    elif args.command == "notes":
        harmonic, percussive = librosa.effects.hpss(signal)

        onset_times = detect_onsets_piano(percussive, sr, delta=args.delta)
        notes = estimate_note_pitches(harmonic, sr, onset_times, frame_duration=args.frame_duration)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(notes, indent=2))
            print(f"Notes saved to: {out}  ({len(notes)} detected)")
        else:
            print(f"{'Time':>8}  {'Note':>5}  {'MIDI':>5}  {'Freq (Hz)':>10}")
            print("-" * 36)
            for n in notes:
                print(f"{n['time']:>8.3f}  {n['note']:>5}  {n['midi']:>5}  {n['freq_hz']:>10.2f}")
            print(f"\n{len(notes)} notes detected.")

    elif args.command == "midi":
        output_path = (
            Path(args.output) if args.output
            else DEFAULT_OUTPUT_DIR / f"{input_path.stem}.mid"
        )
        onset_times = detect_onsets_piano(signal, sr, delta=args.delta)
        notes = estimate_note_pitches(signal, sr, onset_times, frame_duration=args.frame_duration)
        if not notes:
            print("No notes detected — try lowering --delta.")
            return
        notes_to_midi(notes, output_path, tempo=args.tempo)
        print(f"MIDI saved to: {output_path}  ({len(notes)} notes, {args.tempo} BPM)")
    


def _hz_to_note(freq: float) -> str:
    """Best-effort note name for a frequency, used only for CLI display."""
    try:
        import librosa
        midi = round(librosa.hz_to_midi(freq))
        return librosa.midi_to_note(int(midi))
    except Exception:
        return ""


if __name__ == "__main__":
    main()