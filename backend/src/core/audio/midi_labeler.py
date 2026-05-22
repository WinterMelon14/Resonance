"""
midi_labels.py
==============
Convert clean MIDI files <-> (T, 88, 3) label matrices aligned to a CQT grid.

Matrix layout
-------------
  Y[t, i, 0]  active   – 1.0 while note i is held at frame t
  Y[t, i, 1]  onset    – 1.0 only at the first frame of each note
  Y[t, i, 2]  velocity – velocity / 127.0 while note is active, else 0.0

Pitch axis
----------
  index i = midi_pitch - 21   (i=0 → A0=21, i=87 → C8=108)

Time axis
---------
  frame = round(time_seconds * SR / HOP)
  SR  = 44100
  HOP = 512
  fps ≈ 86.13

Inversion loss
--------------
  Sub-frame timing is lost (max ±5.8 ms).  Everything else is exact.
"""

from __future__ import annotations

import numpy as np
import pretty_midi

# ── constants ────────────────────────────────────────────────────────────────
SR: int = 44100
HOP: int = 512
FPS: float = SR / HOP          # ≈ 86.13 frames per second

MIDI_LO: int = 21              # A0
MIDI_HI: int = 108             # C8
N_PITCHES: int = MIDI_HI - MIDI_LO + 1   # 88

# channel indices
CH_ACTIVE: int = 0
CH_ONSET:  int = 1
CH_VEL:    int = 2
N_CH:      int = 3


# ── helpers ──────────────────────────────────────────────────────────────────

def seconds_to_frame(t: float) -> int:
    """Convert a time in seconds to the nearest CQT frame index."""
    return int(round(t * FPS))


def frame_to_seconds(f: int) -> float:
    """Convert a CQT frame index back to its centre time in seconds."""
    return f / FPS


def _pitch_to_idx(pitch: int) -> int | None:
    """Return matrix column for a MIDI pitch, or None if out of range."""
    if MIDI_LO <= pitch <= MIDI_HI:
        return pitch - MIDI_LO
    return None


# ── forward pass: MIDI → matrix ──────────────────────────────────────────────

def midi_to_label_matrix(
    midi_path: str,
    n_frames: int | None = None,
    pad_frames: int = 0,
) -> np.ndarray:
    """
    Load a MIDI file and return a (T, 88, 3) float32 label matrix.

    Parameters
    ----------
    midi_path : str
        Path to the .mid / .midi file.
    n_frames : int, optional
        Force the time axis to exactly this length.  If None, the length is
        derived from the last note-off event.
    pad_frames : int
        Extra frames appended after the last event (only used when n_frames
        is None).

    Returns
    -------
    Y : np.ndarray, shape (T, 88, 3), dtype float32
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Determine matrix length
    if n_frames is not None:
        T = n_frames
    else:
        end_time = pm.get_end_time()
        T = seconds_to_frame(end_time) + 1 + pad_frames

    Y = np.zeros((T, N_PITCHES, N_CH), dtype=np.float32)

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            idx = _pitch_to_idx(note.pitch)
            if idx is None:
                continue

            f_on  = min(seconds_to_frame(note.start), T - 1)
            f_off = min(seconds_to_frame(note.end),   T - 1)

            # When onset and offset land on the same frame, keep at least 1
            if f_off <= f_on:
                f_off = f_on + 1
            f_off = min(f_off, T)          # exclusive upper bound for slice

            vel_norm = note.velocity / 127.0

            Y[f_on:f_off, idx, CH_ACTIVE] = 1.0
            Y[f_on,        idx, CH_ONSET]  = 1.0
            Y[f_on:f_off,  idx, CH_VEL]   = vel_norm
    print(Y)
    return Y


# ── inverse pass: matrix → MIDI ──────────────────────────────────────────────

def label_matrix_to_midi(
    Y: np.ndarray,
    output_path: str | None = None,
    onset_threshold: float = 0.5,
    active_threshold: float = 0.5,
    tempo: float = 120.0,
    program: int = 0,
) -> pretty_midi.PrettyMIDI:
    """
    Reconstruct a PrettyMIDI object from a (T, 88, 3) label matrix.

    The reconstruction algorithm
    ─────────────────────────────
    For each pitch i:
      • A note starts at frame t when onset[t,i] > threshold  (or when
        active[t,i] rises after being 0, as a fallback).
      • The note ends at the first frame where active[t,i] drops to 0,
        or at the last frame T if it never drops.
      • Velocity is the mean of vel[t,i] over the active span, re-scaled
        to 0–127 and clamped to [1, 127].

    Parameters
    ----------
    Y : np.ndarray, shape (T, 88, 3)
    output_path : str, optional
        If given, write the MIDI to this path.
    onset_threshold : float
        Minimum value in CH_ONSET to declare a note start.
    active_threshold : float
        Minimum value in CH_ACTIVE to consider a note held.
    tempo : float
        BPM written into the output file (default 120).
    program : int
        General MIDI program number for the single output instrument.

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
    """
    if Y.ndim != 3 or Y.shape[1] != N_PITCHES or Y.shape[2] != N_CH:
        raise ValueError(f"Expected shape (T, {N_PITCHES}, {N_CH}), got {Y.shape}")

    T = Y.shape[0]
    active  = Y[:, :, CH_ACTIVE] >= active_threshold   # (T, 88) bool
    onset   = Y[:, :, CH_ONSET]  >= onset_threshold    # (T, 88) bool
    vel_mat = Y[:, :, CH_VEL]                           # (T, 88) float

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)

    for i in range(N_PITCHES):
        pitch = i + MIDI_LO
        t = 0
        while t < T:
            # Find a note start: prefer explicit onset, fall back to
            # active rising edge (handles matrices without onset channel).
            if onset[t, i] or (active[t, i] and (t == 0 or not active[t - 1, i])):
                note_start = t
                # Advance until the note ends: stop at active=0 OR a new
                # onset on the same pitch (re-attack without a gap).
                t_end = t + 1
                while t_end < T and active[t_end, i] and not onset[t_end, i]:
                    t_end += 1

                # Mean velocity over the held frames
                vel_raw = vel_mat[note_start:t_end, i].mean()
                velocity = int(np.clip(round(vel_raw * 127), 1, 127))

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=frame_to_seconds(note_start),
                    end=frame_to_seconds(t_end),
                )
                instrument.notes.append(note)
                t = t_end   # jump past this note
            else:
                t += 1

    instrument.notes.sort(key=lambda n: n.start)
    pm.instruments.append(instrument)

    if output_path is not None:
        pm.write(output_path)

    return pm


# ── round-trip verification ───────────────────────────────────────────────────

def verify_roundtrip(
    midi_path: str,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Load a MIDI, convert to matrix, invert back, and report fidelity metrics.

    Returns a dict with keys:
      n_notes_original, n_notes_recovered,
      note_match_rate,
      mean_timing_error_ms, max_timing_error_ms,
      mean_velocity_error, max_velocity_error
    """
    pm_orig = pretty_midi.PrettyMIDI(midi_path)

    # Collect ground-truth notes (in-range pitches only)
    orig_notes = []
    for inst in pm_orig.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if MIDI_LO <= note.pitch <= MIDI_HI:
                orig_notes.append(note)

    # Forward pass
    Y = midi_to_label_matrix(midi_path)

    # Inverse pass
    pm_rec = label_matrix_to_midi(Y, output_path=output_path)
    rec_notes = pm_rec.instruments[0].notes if pm_rec.instruments else []

    # Build lookup: (pitch, approx_frame) → original note
    orig_lookup: dict[tuple[int, int], pretty_midi.Note] = {}
    for note in orig_notes:
        key = (note.pitch, seconds_to_frame(note.start))
        orig_lookup[key] = note

    timing_errors: list[float] = []
    velocity_errors: list[float] = []
    matched = 0

    for note in rec_notes:
        key = (note.pitch, seconds_to_frame(note.start))
        if key in orig_lookup:
            orig = orig_lookup[key]
            matched += 1
            timing_errors.append(abs(note.start - orig.start) * 1000)   # ms
            velocity_errors.append(abs(note.velocity - orig.velocity))

    n_orig = len(orig_notes)
    n_rec  = len(rec_notes)

    metrics = {
        "n_notes_original":    n_orig,
        "n_notes_recovered":   n_rec,
        "note_match_rate":     matched / n_orig if n_orig else 0.0,
        "mean_timing_error_ms": float(np.mean(timing_errors))   if timing_errors else 0.0,
        "max_timing_error_ms":  float(np.max(timing_errors))    if timing_errors else 0.0,
        "mean_velocity_error":  float(np.mean(velocity_errors)) if velocity_errors else 0.0,
        "max_velocity_error":   float(np.max(velocity_errors))  if velocity_errors else 0.0,
    }

    if verbose:
        print("── Round-trip verification ──────────────────────────")
        print(f"  Notes original   : {n_orig}")
        print(f"  Notes recovered  : {n_rec}")
        print(f"  Match rate       : {metrics['note_match_rate']:.1%}")
        print(f"  Timing error     : mean {metrics['mean_timing_error_ms']:.2f} ms"
              f"  /  max {metrics['max_timing_error_ms']:.2f} ms")
        print(f"  Velocity error   : mean {metrics['mean_velocity_error']:.2f}"
              f"  /  max {metrics['max_velocity_error']:.0f}")
        print("─────────────────────────────────────────────────────")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="MIDI ↔ label-matrix pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # encode
    enc = sub.add_parser("encode", help="MIDI → .npy label matrix")
    enc.add_argument("midi",   help="Input .mid file")
    enc.add_argument("output", help="Output .npy file")
    enc.add_argument("--n-frames", type=int, default=None)
    enc.add_argument("--pad",      type=int, default=0)

    # decode
    dec = sub.add_parser("decode", help=".npy label matrix → MIDI")
    dec.add_argument("npy",    help="Input .npy file  (T, 88, 3)")
    dec.add_argument("output", help="Output .mid file")
    dec.add_argument("--tempo",   type=float, default=120.0)
    dec.add_argument("--program", type=int,   default=0)

    # verify
    ver = sub.add_parser("verify", help="Round-trip fidelity check")
    ver.add_argument("midi",             help="Input .mid file")
    ver.add_argument("--output", default=None, help="Save recovered MIDI here")

    args = parser.parse_args()

    if args.cmd == "encode":
        Y = midi_to_label_matrix(args.midi, n_frames=args.n_frames, pad_frames=args.pad)
        np.save(args.output, Y)
        print(f"Saved {Y.shape} matrix → {args.output}")

    elif args.cmd == "decode":
        Y = np.load(args.npy)
        label_matrix_to_midi(Y, output_path=args.output,
                             tempo=args.tempo, program=args.program)
        print(f"Saved MIDI → {args.output}")

    elif args.cmd == "verify":
        verify_roundtrip(args.midi, output_path=args.output)

    else:
        parser.print_help()
        sys.exit(1)