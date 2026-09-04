"""Compare MIDI clean-up versions and evaluate MIDI against ground truth.

Examples
--------
Compare a raw transcription with its cleaned version::

    python -m backend.midi_compare diff raw.mid cleaned.mid

Compare both versions with a ground-truth MIDI::

    python -m backend.midi_compare diff raw.mid cleaned.mid --truth truth.mid

Evaluate one prediction directly::

    python -m backend.midi_compare evaluate prediction.mid truth.mid

Comparisons flatten all non-drum instruments and match notes by pitch and time.
Tempo-map events do not need to match because PrettyMIDI exposes note times in
seconds after applying each file's tempo map.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean, median
from typing import Sequence

import pretty_midi


@dataclass(frozen=True, slots=True)
class MidiNote:
    """A normalized non-drum MIDI note used for comparisons."""

    pitch: int
    start: float
    end: float
    velocity: int
    instrument: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_midi_notes(midi_path: str | Path) -> list[MidiNote]:
    """Load and flatten all valid, non-drum notes from a MIDI file."""
    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[MidiNote] = []
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.end <= note.start:
                continue
            notes.append(
                MidiNote(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    velocity=int(note.velocity),
                    instrument=instrument_index,
                )
            )

    return sorted(notes, key=lambda note: (note.pitch, note.start, note.end))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _error_summary(values: list[float]) -> dict[str, float | int | None]:
    absolute = [abs(value) for value in values]
    return {
        "count": len(values),
        "mean_signed_ms": 1000.0 * fmean(values) if values else None,
        "mean_absolute_ms": 1000.0 * fmean(absolute) if absolute else None,
        "median_absolute_ms": 1000.0 * median(absolute) if absolute else None,
        "p95_absolute_ms": (
            1000.0 * _percentile(absolute, 0.95) if absolute else None
        ),
        "max_absolute_ms": 1000.0 * max(absolute) if absolute else None,
    }


def summarize_midi_notes(
    notes: list[MidiNote],
    *,
    short_note_seconds: float = 0.125,
    tiny_gap_seconds: float = 0.050,
) -> dict[str, float | int | None]:
    """Report notation-cleanliness indicators without requiring ground truth."""
    durations = [note.duration for note in notes]
    by_pitch: dict[int, list[MidiNote]] = defaultdict(list)
    for note in notes:
        by_pitch[note.pitch].append(note)

    same_pitch_overlaps = 0
    tiny_same_pitch_gaps = 0
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda note: (note.start, note.end))
        previous_end: float | None = None
        for note in pitch_notes:
            if previous_end is not None:
                gap = note.start - previous_end
                if gap < 0:
                    same_pitch_overlaps += 1
                elif gap <= tiny_gap_seconds:
                    tiny_same_pitch_gaps += 1
            previous_end = (
                note.end if previous_end is None else max(previous_end, note.end)
            )

    return {
        "note_count": len(notes),
        "unique_pitches": len(by_pitch),
        "first_onset_seconds": min((note.start for note in notes), default=None),
        "last_offset_seconds": max((note.end for note in notes), default=None),
        "mean_duration_ms": 1000.0 * fmean(durations) if durations else None,
        "median_duration_ms": 1000.0 * median(durations) if durations else None,
        "minimum_duration_ms": 1000.0 * min(durations) if durations else None,
        "short_note_count": sum(
            note.duration < short_note_seconds for note in notes
        ),
        "same_pitch_overlap_count": same_pitch_overlaps,
        "tiny_same_pitch_gap_count": tiny_same_pitch_gaps,
    }


def _group_by_pitch(notes: list[MidiNote]) -> dict[int, list[MidiNote]]:
    grouped: dict[int, list[MidiNote]] = defaultdict(list)
    for note in notes:
        grouped[note.pitch].append(note)
    for pitch_notes in grouped.values():
        pitch_notes.sort(key=lambda note: (note.start, note.end))
    return grouped


def _match_notes_by_onset(
    first: list[MidiNote],
    second: list[MidiNote],
    onset_tolerance_seconds: float,
) -> tuple[list[tuple[MidiNote, MidiNote]], list[MidiNote], list[MidiNote]]:
    """Greedily make ordered, one-to-one matches for equal-pitch notes.

    For a single pitch, note order is preserved. This avoids matching a later
    repetition to an earlier occurrence when several repeated notes are close.
    """
    first_by_pitch = _group_by_pitch(first)
    second_by_pitch = _group_by_pitch(second)
    matches: list[tuple[MidiNote, MidiNote]] = []
    unmatched_first: list[MidiNote] = []
    unmatched_second: list[MidiNote] = []

    for pitch in sorted(set(first_by_pitch) | set(second_by_pitch)):
        left = first_by_pitch.get(pitch, [])
        right = second_by_pitch.get(pitch, [])
        left_index = right_index = 0

        while left_index < len(left) and right_index < len(right):
            left_note = left[left_index]
            right_note = right[right_index]
            difference = left_note.start - right_note.start

            if abs(difference) <= onset_tolerance_seconds:
                matches.append((left_note, right_note))
                left_index += 1
                right_index += 1
            elif difference < 0:
                unmatched_first.append(left_note)
                left_index += 1
            else:
                unmatched_second.append(right_note)
                right_index += 1

        unmatched_first.extend(left[left_index:])
        unmatched_second.extend(right[right_index:])

    matches.sort(key=lambda pair: (pair[0].start, pair[0].pitch))
    unmatched_first.sort(key=lambda note: (note.start, note.pitch))
    unmatched_second.sort(key=lambda note: (note.start, note.pitch))
    return matches, unmatched_first, unmatched_second


def _note_dict(note: MidiNote) -> dict[str, float | int]:
    result = asdict(note)
    result["duration"] = note.duration
    return result


def _metrics(tp: int, predicted_count: int, reference_count: int) -> dict:
    fp = predicted_count - tp
    fn = reference_count - tp
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / reference_count if reference_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_midi_files(
    before_path: str | Path,
    after_path: str | Path,
    *,
    match_tolerance_seconds: float = 0.010,
    change_epsilon_seconds: float = 0.001,
    short_note_seconds: float = 0.125,
    tiny_gap_seconds: float = 0.050,
    max_examples: int = 20,
) -> dict:
    """Describe changes between raw and cleaned MIDI files.

    This is a diagnostic diff, not an accuracy metric. Notes match when their
    pitches agree and their onsets are within ``match_tolerance_seconds``.
    """
    before_path = Path(before_path)
    after_path = Path(after_path)
    before = load_midi_notes(before_path)
    after = load_midi_notes(after_path)
    matches, removed, added = _match_notes_by_onset(
        before,
        after,
        match_tolerance_seconds,
    )

    onset_errors = [new.start - old.start for old, new in matches]
    offset_errors = [new.end - old.end for old, new in matches]
    duration_errors = [new.duration - old.duration for old, new in matches]
    changed_onsets = [
        (old, new)
        for old, new in matches
        if abs(new.start - old.start) > change_epsilon_seconds
    ]
    changed_offsets = [
        (old, new)
        for old, new in matches
        if abs(new.end - old.end) > change_epsilon_seconds
    ]
    changed_velocities = [
        (old, new) for old, new in matches if old.velocity != new.velocity
    ]

    before_counts = Counter(note.pitch for note in before)
    after_counts = Counter(note.pitch for note in after)
    pitch_count_delta = {
        str(pitch): after_counts[pitch] - before_counts[pitch]
        for pitch in sorted(set(before_counts) | set(after_counts))
        if after_counts[pitch] != before_counts[pitch]
    }

    before_summary = summarize_midi_notes(
        before,
        short_note_seconds=short_note_seconds,
        tiny_gap_seconds=tiny_gap_seconds,
    )
    after_summary = summarize_midi_notes(
        after,
        short_note_seconds=short_note_seconds,
        tiny_gap_seconds=tiny_gap_seconds,
    )

    return {
        "settings": {
            "match_tolerance_ms": match_tolerance_seconds * 1000.0,
            "change_epsilon_ms": change_epsilon_seconds * 1000.0,
            "short_note_ms": short_note_seconds * 1000.0,
            "tiny_gap_ms": tiny_gap_seconds * 1000.0,
        },
        "files": {
            "before": str(before_path),
            "after": str(after_path),
        },
        "before": before_summary,
        "after": after_summary,
        "changes": {
            "matched_notes": len(matches),
            "removed_notes": len(removed),
            "added_notes": len(added),
            "onset_changed_notes": len(changed_onsets),
            "offset_changed_notes": len(changed_offsets),
            "velocity_changed_notes": len(changed_velocities),
            "onset_shift": _error_summary(onset_errors),
            "offset_shift": _error_summary(offset_errors),
            "duration_change": _error_summary(duration_errors),
            "pitch_note_count_delta": pitch_count_delta,
        },
        "examples": {
            "removed": [_note_dict(note) for note in removed[:max_examples]],
            "added": [_note_dict(note) for note in added[:max_examples]],
            "changed_offsets": [
                {
                    "before": _note_dict(old),
                    "after": _note_dict(new),
                    "offset_change_ms": 1000.0 * (new.end - old.end),
                }
                for old, new in changed_offsets[:max_examples]
            ],
        },
    }


def evaluate_midi_against_truth(
    predicted_path: str | Path,
    truth_path: str | Path,
    *,
    onset_tolerance_seconds: float = 0.050,
    minimum_offset_tolerance_seconds: float = 0.050,
    offset_tolerance_ratio: float = 0.20,
    align_first_onset: bool = False,
    short_note_seconds: float = 0.125,
    tiny_gap_seconds: float = 0.050,
    max_examples: int = 20,
) -> dict:
    """Evaluate a predicted MIDI against user-provided ground truth.

    Onset matches require equal pitch and onset distance within the configured
    tolerance. An onset+offset match additionally requires the offset error to
    be no more than the greater of the fixed tolerance and a fraction of the
    truth note's duration.
    """
    predicted_path = Path(predicted_path)
    truth_path = Path(truth_path)
    predicted = load_midi_notes(predicted_path)
    truth = load_midi_notes(truth_path)

    alignment_shift = 0.0
    if align_first_onset and predicted and truth:
        alignment_shift = min(note.start for note in truth) - min(
            note.start for note in predicted
        )
        predicted = [
            replace(
                note,
                start=note.start + alignment_shift,
                end=note.end + alignment_shift,
            )
            for note in predicted
        ]

    matches, false_positives, false_negatives = _match_notes_by_onset(
        predicted,
        truth,
        onset_tolerance_seconds,
    )
    onset_tp = len(matches)
    onset_offset_matches: list[tuple[MidiNote, MidiNote]] = []
    onset_errors: list[float] = []
    offset_errors: list[float] = []
    duration_errors: list[float] = []

    for predicted_note, truth_note in matches:
        onset_errors.append(predicted_note.start - truth_note.start)
        offset_errors.append(predicted_note.end - truth_note.end)
        duration_errors.append(predicted_note.duration - truth_note.duration)
        allowed_offset_error = max(
            minimum_offset_tolerance_seconds,
            offset_tolerance_ratio * truth_note.duration,
        )
        if abs(predicted_note.end - truth_note.end) <= allowed_offset_error:
            onset_offset_matches.append((predicted_note, truth_note))

    return {
        "settings": {
            "onset_tolerance_ms": onset_tolerance_seconds * 1000.0,
            "minimum_offset_tolerance_ms": (
                minimum_offset_tolerance_seconds * 1000.0
            ),
            "offset_tolerance_ratio": offset_tolerance_ratio,
            "align_first_onset": align_first_onset,
            "alignment_shift_ms": alignment_shift * 1000.0,
        },
        "files": {
            "predicted": str(predicted_path),
            "truth": str(truth_path),
        },
        "predicted": summarize_midi_notes(
            predicted,
            short_note_seconds=short_note_seconds,
            tiny_gap_seconds=tiny_gap_seconds,
        ),
        "truth": summarize_midi_notes(
            truth,
            short_note_seconds=short_note_seconds,
            tiny_gap_seconds=tiny_gap_seconds,
        ),
        "metrics": {
            "onset": _metrics(onset_tp, len(predicted), len(truth)),
            "onset_offset": _metrics(
                len(onset_offset_matches),
                len(predicted),
                len(truth),
            ),
            "onset_error": _error_summary(onset_errors),
            "offset_error_for_onset_matches": _error_summary(offset_errors),
            "duration_error_for_onset_matches": _error_summary(duration_errors),
        },
        "examples": {
            "false_positives": [
                _note_dict(note) for note in false_positives[:max_examples]
            ],
            "false_negatives": [
                _note_dict(note) for note in false_negatives[:max_examples]
            ],
        },
    }


def compare_midi_versions(
    before_path: str | Path,
    after_path: str | Path,
    *,
    ground_truth_path: str | Path | None = None,
    match_tolerance_seconds: float = 0.010,
    onset_tolerance_seconds: float = 0.050,
    minimum_offset_tolerance_seconds: float = 0.050,
    offset_tolerance_ratio: float = 0.20,
    align_first_onset: bool = False,
    short_note_seconds: float = 0.125,
    tiny_gap_seconds: float = 0.050,
    max_examples: int = 20,
) -> dict:
    """Compare before/after MIDI and optionally score both against truth."""
    report = {
        "before_after": compare_midi_files(
            before_path,
            after_path,
            match_tolerance_seconds=match_tolerance_seconds,
            short_note_seconds=short_note_seconds,
            tiny_gap_seconds=tiny_gap_seconds,
            max_examples=max_examples,
        )
    }
    if ground_truth_path is None:
        return report

    evaluation_options = {
        "onset_tolerance_seconds": onset_tolerance_seconds,
        "minimum_offset_tolerance_seconds": minimum_offset_tolerance_seconds,
        "offset_tolerance_ratio": offset_tolerance_ratio,
        "align_first_onset": align_first_onset,
        "short_note_seconds": short_note_seconds,
        "tiny_gap_seconds": tiny_gap_seconds,
        "max_examples": max_examples,
    }
    before_evaluation = evaluate_midi_against_truth(
        before_path,
        ground_truth_path,
        **evaluation_options,
    )
    after_evaluation = evaluate_midi_against_truth(
        after_path,
        ground_truth_path,
        **evaluation_options,
    )
    before_metrics = before_evaluation["metrics"]
    after_metrics = after_evaluation["metrics"]
    report["ground_truth_comparison"] = {
        "before": before_evaluation,
        "after": after_evaluation,
        "delta": {
            "onset_precision": (
                after_metrics["onset"]["precision"]
                - before_metrics["onset"]["precision"]
            ),
            "onset_recall": (
                after_metrics["onset"]["recall"]
                - before_metrics["onset"]["recall"]
            ),
            "onset_f1": (
                after_metrics["onset"]["f1"]
                - before_metrics["onset"]["f1"]
            ),
            "onset_offset_f1": (
                after_metrics["onset_offset"]["f1"]
                - before_metrics["onset_offset"]["f1"]
            ),
        },
    }
    return report


def write_report(report: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _print_diff(report: dict) -> None:
    before_after = report["before_after"]
    before = before_after["before"]
    after = before_after["after"]
    changes = before_after["changes"]
    print("MIDI cleanup comparison")
    print(f"  Notes:              {before['note_count']} -> {after['note_count']}")
    print(
        "  Short notes:        "
        f"{before['short_note_count']} -> {after['short_note_count']}"
    )
    print(
        "  Same-pitch overlap: "
        f"{before['same_pitch_overlap_count']} -> "
        f"{after['same_pitch_overlap_count']}"
    )
    print(
        "  Tiny same-pitch gap: "
        f"{before['tiny_same_pitch_gap_count']} -> "
        f"{after['tiny_same_pitch_gap_count']}"
    )
    print(
        "  Matched/removed/added: "
        f"{changes['matched_notes']} / {changes['removed_notes']} / "
        f"{changes['added_notes']}"
    )
    print(f"  Changed offsets:    {changes['offset_changed_notes']}")
    print(
        "  Mean |offset shift|: "
        f"{_format_optional(changes['offset_shift']['mean_absolute_ms'], ' ms')}"
    )

    comparison = report.get("ground_truth_comparison")
    if comparison:
        before_metrics = comparison["before"]["metrics"]
        after_metrics = comparison["after"]["metrics"]
        delta = comparison["delta"]
        print("Ground-truth result")
        print(
            "  Onset F1:          "
            f"{before_metrics['onset']['f1']:.4f} -> "
            f"{after_metrics['onset']['f1']:.4f} "
            f"({delta['onset_f1']:+.4f})"
        )
        print(
            "  Onset+offset F1:   "
            f"{before_metrics['onset_offset']['f1']:.4f} -> "
            f"{after_metrics['onset_offset']['f1']:.4f} "
            f"({delta['onset_offset_f1']:+.4f})"
        )


def _print_evaluation(report: dict) -> None:
    onset = report["metrics"]["onset"]
    onset_offset = report["metrics"]["onset_offset"]
    print("MIDI ground-truth evaluation")
    print(
        f"  Predicted/truth notes: {report['predicted']['note_count']} / "
        f"{report['truth']['note_count']}"
    )
    print(
        "  Onset P/R/F1:         "
        f"{onset['precision']:.4f} / {onset['recall']:.4f} / {onset['f1']:.4f}"
    )
    print(
        "  Onset+offset P/R/F1:  "
        f"{onset_offset['precision']:.4f} / "
        f"{onset_offset['recall']:.4f} / {onset_offset['f1']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare MIDI versions or evaluate MIDI against ground truth."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare a before MIDI with an after MIDI",
    )
    diff_parser.add_argument("before", type=Path)
    diff_parser.add_argument("after", type=Path)
    diff_parser.add_argument("--truth", type=Path, default=None)
    diff_parser.add_argument("--match-tolerance-ms", type=float, default=10.0)
    diff_parser.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    diff_parser.add_argument("--offset-tolerance-ms", type=float, default=50.0)
    diff_parser.add_argument("--offset-tolerance-ratio", type=float, default=0.20)
    diff_parser.add_argument("--short-note-ms", type=float, default=125.0)
    diff_parser.add_argument("--tiny-gap-ms", type=float, default=50.0)
    diff_parser.add_argument("--align-first-onset", action="store_true")
    diff_parser.add_argument("--max-examples", type=int, default=20)
    diff_parser.add_argument("--json", type=Path, default=None)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one predicted MIDI against ground truth",
    )
    evaluate_parser.add_argument("predicted", type=Path)
    evaluate_parser.add_argument("truth", type=Path)
    evaluate_parser.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    evaluate_parser.add_argument("--offset-tolerance-ms", type=float, default=50.0)
    evaluate_parser.add_argument(
        "--offset-tolerance-ratio",
        type=float,
        default=0.20,
    )
    evaluate_parser.add_argument("--short-note-ms", type=float, default=125.0)
    evaluate_parser.add_argument("--tiny-gap-ms", type=float, default=50.0)
    evaluate_parser.add_argument("--align-first-onset", action="store_true")
    evaluate_parser.add_argument("--max-examples", type=int, default=20)
    evaluate_parser.add_argument("--json", type=Path, default=None)
    return parser


def _positive_milliseconds(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value / 1000.0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    short_note_seconds = _positive_milliseconds(
        args.short_note_ms,
        "--short-note-ms",
    )
    tiny_gap_seconds = _positive_milliseconds(
        args.tiny_gap_ms,
        "--tiny-gap-ms",
    )
    if args.offset_tolerance_ratio < 0:
        raise ValueError("--offset-tolerance-ratio must be non-negative.")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative.")

    if args.command == "diff":
        report = compare_midi_versions(
            args.before,
            args.after,
            ground_truth_path=args.truth,
            match_tolerance_seconds=_positive_milliseconds(
                args.match_tolerance_ms,
                "--match-tolerance-ms",
            ),
            onset_tolerance_seconds=_positive_milliseconds(
                args.onset_tolerance_ms,
                "--onset-tolerance-ms",
            ),
            minimum_offset_tolerance_seconds=_positive_milliseconds(
                args.offset_tolerance_ms,
                "--offset-tolerance-ms",
            ),
            offset_tolerance_ratio=args.offset_tolerance_ratio,
            align_first_onset=args.align_first_onset,
            short_note_seconds=short_note_seconds,
            tiny_gap_seconds=tiny_gap_seconds,
            max_examples=args.max_examples,
        )
        _print_diff(report)
    elif args.command == "evaluate":
        report = evaluate_midi_against_truth(
            args.predicted,
            args.truth,
            onset_tolerance_seconds=_positive_milliseconds(
                args.onset_tolerance_ms,
                "--onset-tolerance-ms",
            ),
            minimum_offset_tolerance_seconds=_positive_milliseconds(
                args.offset_tolerance_ms,
                "--offset-tolerance-ms",
            ),
            offset_tolerance_ratio=args.offset_tolerance_ratio,
            align_first_onset=args.align_first_onset,
            short_note_seconds=short_note_seconds,
            tiny_gap_seconds=tiny_gap_seconds,
            max_examples=args.max_examples,
        )
        _print_evaluation(report)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    if args.json is not None:
        output_path = write_report(report, args.json)
        print(f"JSON report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())