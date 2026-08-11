from __future__ import annotations

import json
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import torch
from tqdm.auto import tqdm

from backend.config import CFG, DecoderConfig, ExperimentConfig, FeatureConfig


CH_ACTIVE = 0
CH_ONSET = 1
CH_OFFSET = 2
N_LABEL_CHANNELS = 3

END_REASON_NONE = 0
END_REASON_REONSET = 1
END_REASON_ACTIVE_DROPOUT = 2
END_REASON_GATED_OFFSET = 3

DEFAULT_CHUNK_FRAMES = 384
DEFAULT_INFERENCE_BATCH_SIZE = 8

# These functions are all straight out of the notebook
def default_threshold_results_path(cfg: ExperimentConfig = CFG) -> Path:
    return Path(cfg.paths.checkpoint_path).parent / "threshold_results.csv"


def apply_best_decoder_thresholds(
    cfg: ExperimentConfig = CFG,
    threshold_results_path: str | Path | None = None,
) -> dict[str, float]:
    # Change cfg.decoder to the thresholds we found
    path = (
        default_threshold_results_path(cfg)
        if threshold_results_path is None
        else Path(threshold_results_path)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Decoder threshold results were not found at {path}. "
            "Pass threshold_results_path explicitly or place threshold_results.csv "
            "beside the configured checkpoint."
        )

    threshold_results = pd.read_csv(path)
    if threshold_results.empty:
        raise ValueError(f"Decoder threshold results are empty: {path}")

    column_to_field = {
        "active_threshold": "active_start_threshold",
        "continue_threshold": "active_continue_threshold",
        "onset_threshold": "onset_threshold",
        "offset_threshold": "offset_threshold",
    }
    missing = [column for column in column_to_field if column not in threshold_results]
    if missing:
        raise ValueError(
            f"{path} is missing required threshold columns: {', '.join(missing)}"
        )

    best_row = threshold_results.iloc[0]
    selected: dict[str, float] = {}
    for column, field_name in column_to_field.items():
        value = float(best_row[column])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {column}={value!r} in {path}")
        setattr(cfg.decoder, field_name, value)
        selected[field_name] = value

    if cfg.decoder.active_continue_threshold > cfg.decoder.active_start_threshold:
        raise ValueError(
            "The best threshold row has continue_threshold greater than "
            "active_threshold."
        )

    if "score" in best_row:
        selected["score"] = float(best_row["score"])
    return selected


def _normalize_db_values(db: np.ndarray) -> np.ndarray:
    lo, hi = float(db.min()), float(db.max())
    return ((db - lo) / (hi - lo + 1e-8)).astype(np.float32)


# Normalizes amplitude to dB and scales to [0, 1]
def normalize_db_feature(magnitude: np.ndarray) -> np.ndarray:
    db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return _normalize_db_values(db)

# Some tunes may not be in 440hz 
def estimate_concert_pitch(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[float, float]:
    harmonic_audio = librosa.effects.harmonic(audio)

    tuning_semitones = float(
        librosa.estimate_tuning(
            y=harmonic_audio,
            sr=sample_rate,
            bins_per_octave=12,
            resolution=0.01,
            fmin=librosa.note_to_hz("A0"),
            fmax=librosa.note_to_hz("C8"),
        )
    )

    cents = tuning_semitones * 100.0
    concert_pitch_hz = 440.0 * 2 ** (tuning_semitones / 12.0)

    return concert_pitch_hz, cents

# There is a logmel option for future testing in case I want to use it instead
# of CQT, but the trained baseline expects CQT.
def compute_features(
    audio_path: str | Path,
    feature_cfg: FeatureConfig = CFG.feature,
    concert_pitch_hz: float | Literal["auto"] = "auto",
) -> np.ndarray:
    """Return a time-first feature matrix with shape (f,b)."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, _ = librosa.load(audio_path, sr=feature_cfg.sample_rate, mono=True)

    if concert_pitch_hz == "auto":
        concert_pitch_hz, tuning_cents = estimate_concert_pitch(
            y,
            feature_cfg.sample_rate,
        )
        print(
            f"Estimated tuning: A4={concert_pitch_hz:.2f} Hz "
            f"({tuning_cents:+.1f} cents)"
        )
    else:
        concert_pitch_hz = float(concert_pitch_hz)

    fmin_hz = (
        float(librosa.note_to_hz("A0"))
        * concert_pitch_hz
        / 440.0
    )


    if len(y) == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    if feature_cfg.kind == "cqt":
        transform = librosa.cqt(
            y,
            sr=feature_cfg.sample_rate,
            hop_length=feature_cfg.hop_length,
            fmin=fmin_hz,
            n_bins=feature_cfg.cqt_extract_bins,
            bins_per_octave=feature_cfg.cqt_bins_per_octave,
        )
        feature = normalize_db_feature(np.abs(transform).astype(np.float32))
    elif feature_cfg.kind == "log_mel":
        power = librosa.feature.melspectrogram(
            y=y,
            sr=feature_cfg.sample_rate,
            hop_length=feature_cfg.hop_length,
            n_mels=feature_cfg.n_bins,
            fmin=float(librosa.note_to_hz("A0")),
            power=2.0,
        )
        feature = _normalize_db_values(librosa.power_to_db(power, ref=np.max))
    else:
        raise ValueError(f"Unknown feature kind: {feature_cfg.kind}")

    return feature.T


def binary_counts(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {pred.shape} vs {truth.shape}")
    return np.array(
        [
            np.logical_and(pred, truth).sum(),
            np.logical_and(pred, ~truth).sum(),
            np.logical_and(~pred, truth).sum(),
        ],
        dtype=np.int64,
    )


def counts_to_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def onset_event_counts(
    pred: np.ndarray,
    truth: np.ndarray,
    tolerance_frames: int,
) -> np.ndarray:
    """Greedily match predicted events to truth by pitch within a tolerance."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(
            f"Expected matching (frames, pitches) event rolls, got {pred.shape} "
            f"and {truth.shape}."
        )

    tp = fp = fn = 0
    for pitch in range(pred.shape[1]):
        predicted_frames = np.flatnonzero(pred[:, pitch])
        true_frames = np.flatnonzero(truth[:, pitch])
        used = np.zeros(len(true_frames), dtype=bool)

        for predicted_frame in predicted_frames:
            candidates = np.flatnonzero(
                (~used) & (np.abs(true_frames - predicted_frame) <= tolerance_frames)
            )
            if len(candidates) == 0:
                fp += 1
                continue
            distances = np.abs(true_frames[candidates] - predicted_frame)
            match = int(candidates[np.argmin(distances)])
            used[match] = True
            tp += 1

        fn += int((~used).sum())

    return np.array([tp, fp, fn], dtype=np.int64)


def peak_pick_onsets(
    probabilities: np.ndarray,
    threshold: float,
    pre_max: int,
    post_max: int,
    min_gap_frames: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probabilities with shape (frames, pitches), got {probabilities.shape}"
        )

    frames, pitches = probabilities.shape
    peaks = np.zeros((frames, pitches), dtype=bool)
    for pitch in range(pitches):
        candidates = []
        for frame in np.flatnonzero(probabilities[:, pitch] >= threshold):
            left = max(0, frame - pre_max)
            right = min(frames, frame + post_max + 1)
            if probabilities[frame, pitch] >= probabilities[left:right, pitch].max():
                candidates.append(int(frame))

        last_kept = -min_gap_frames
        for frame in candidates:
            if frame - last_kept >= min_gap_frames:
                peaks[frame, pitch] = True
                last_kept = frame
            elif probabilities[frame, pitch] > probabilities[last_kept, pitch]:
                peaks[last_kept, pitch] = False
                peaks[frame, pitch] = True
                last_kept = frame
    return peaks


# Use our super accurate active head to see if there's active right after this
# current point. If so then we probably have an onset here.
def active_support_near_onset(
    active_prob: np.ndarray,
    onset_binary: np.ndarray,
    threshold: float,
    radius: int,
) -> np.ndarray:
    keep = np.zeros_like(onset_binary, dtype=bool)
    total_frames = len(active_prob)
    for frame, pitch in zip(*np.where(onset_binary)):
        left = max(0, frame - radius)
        right = min(total_frames, frame + radius + 1)
        keep[frame, pitch] = active_prob[left:right, pitch].max() >= threshold
    return keep


def decode_active_from_onsets(
    active_prob: np.ndarray,
    onset_binary: np.ndarray,
    offset_binary: np.ndarray | None,
    decoder_cfg: DecoderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode notes with the calibrated active head as duration authority."""
    if active_prob.shape != onset_binary.shape:
        raise ValueError(
            f"active/onset shape mismatch: {active_prob.shape} vs {onset_binary.shape}"
        )
    if offset_binary is not None and offset_binary.shape != active_prob.shape:
        raise ValueError(
            f"active/offset shape mismatch: {active_prob.shape} vs {offset_binary.shape}"
        )
    if not (
        0
        <= decoder_cfg.active_end_threshold
        <= decoder_cfg.active_continue_threshold
        <= decoder_cfg.active_start_threshold
        <= 1
    ):
        raise ValueError(
            "Expected active_end_threshold <= active_continue_threshold "
            "<= active_start_threshold, all within [0, 1]."
        )
    if decoder_cfg.active_end_patience_frames < 1:
        raise ValueError("active_end_patience_frames must be at least 1")

    total_frames, pitches = active_prob.shape
    active_out = np.zeros((total_frames, pitches), np.float32)
    onset_out = np.zeros((total_frames, pitches), np.float32)
    offset_out = np.zeros((total_frames, pitches), np.float32)
    end_reason_out = np.zeros((total_frames, pitches), np.uint8)

    candidates_by_pitch: dict[int, list[int]] = {}
    for frame, pitch in zip(*np.where(onset_binary)):
        candidates_by_pitch.setdefault(int(pitch), []).append(int(frame))

    for pitch, candidate_frames in candidates_by_pitch.items():
        candidate_frames.sort()

        # Reject unsupported onset peaks before determining re-onset boundaries.
        frames = []
        for start in candidate_frames:
            left = max(0, start - decoder_cfg.active_support_radius)
            right = min(total_frames, start + decoder_cfg.active_support_radius + 1)
            local_max = active_prob[left:right, pitch].max()
            if (
                active_prob[start, pitch] >= decoder_cfg.active_start_threshold
                or local_max >= decoder_cfg.active_start_threshold
            ):
                frames.append(start)

        for index, start in enumerate(frames):
            next_onset_boundary = (
                frames[index + 1] if index + 1 < len(frames) else total_frames
            )
            minimum_end = min(start + decoder_cfg.min_note_frames, total_frames)

            active_boundary = next_onset_boundary
            low_run = 0
            for frame in range(minimum_end, next_onset_boundary):
                probability = active_prob[frame, pitch]
                if probability >= decoder_cfg.active_continue_threshold:
                    low_run = 0
                elif probability < decoder_cfg.active_end_threshold:
                    low_run += 1
                    if low_run >= decoder_cfg.active_end_patience_frames:
                        active_boundary = (
                            frame - decoder_cfg.active_end_patience_frames + 1
                        )
                        break
                else:
                    low_run = 0

            # Supporting duration signal: accept only offset peaks accompanied
            # by a sufficiently low active value or a meaningful downward move.
            offset_boundary = next_onset_boundary
            if offset_binary is not None and minimum_end < next_onset_boundary:
                hits = np.flatnonzero(
                    offset_binary[minimum_end:next_onset_boundary, pitch]
                )
                for relative_hit in hits:
                    hit = minimum_end + int(relative_hit)
                    peak_left = max(
                        start, hit - decoder_cfg.offset_peak_window_frames
                    )
                    recent_peak = float(active_prob[peak_left : hit + 1, pitch].max())
                    post_right = min(
                        next_onset_boundary,
                        hit + decoder_cfg.offset_lookahead_frames + 1,
                    )
                    post_min = float(active_prob[hit:post_right, pitch].min())
                    active_drop = recent_peak - post_min

                    strongly_inactive = post_min < decoder_cfg.active_end_threshold
                    falling_below_continue = (
                        post_min < decoder_cfg.active_continue_threshold
                        and active_drop >= decoder_cfg.offset_active_drop
                    )
                    if strongly_inactive or falling_below_continue:
                        offset_boundary = hit
                        break

            end = min(next_onset_boundary, active_boundary, offset_boundary)
            if end - start < decoder_cfg.min_note_frames:
                continue

            active_out[start:end, pitch] = 1
            onset_out[start, pitch] = 1
            if end < total_frames:
                offset_out[end, pitch] = 1
                if end == next_onset_boundary:
                    end_reason_out[end, pitch] = END_REASON_REONSET
                elif end == offset_boundary and offset_boundary <= active_boundary:
                    end_reason_out[end, pitch] = END_REASON_GATED_OFFSET
                else:
                    end_reason_out[end, pitch] = END_REASON_ACTIVE_DROPOUT

    return active_out, onset_out, offset_out, end_reason_out


def _decode_probabilities_with_config(
    active_prob: np.ndarray,
    onset_prob: np.ndarray,
    offset_prob: np.ndarray,
    decoder_cfg: DecoderConfig,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    onset_binary = peak_pick_onsets(
        onset_prob,
        decoder_cfg.onset_threshold,
        decoder_cfg.onset_pre_max,
        decoder_cfg.onset_post_max,
        decoder_cfg.onset_min_gap_frames,
    )
    onset_binary = active_support_near_onset(
        active_prob,
        onset_binary,
        threshold=decoder_cfg.active_support_threshold,
        radius=decoder_cfg.active_support_radius,
    )

    offset_binary = peak_pick_onsets(
        offset_prob,
        decoder_cfg.offset_threshold,
        decoder_cfg.offset_pre_max,
        decoder_cfg.offset_post_max,
        decoder_cfg.offset_min_gap_frames,
    )
    active, onset, decoded_offset, end_reason = decode_active_from_onsets(
        active_prob,
        onset_binary,
        offset_binary if decoder_cfg.use_offset_head else None,
        decoder_cfg,
    )
    return {
        "active": active,
        "onset": onset,
        "decoded_offset": decoded_offset,
        "diagnostics": {
            "accepted_onsets": onset_binary,
            "raw_offset_peaks": offset_binary,
            "end_reason": end_reason,
        },
    }

# Decoder configured from the best saved threshold sweep row
class PianoDecoder:

    def __init__(
        self,
        cfg: ExperimentConfig = CFG,
        threshold_results_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.threshold_results_path = (
            default_threshold_results_path(cfg)
            if threshold_results_path is None
            else Path(threshold_results_path)
        )
        # Required invariant: CFG is updated from threshold_results.iloc[0]
        # before this decoder is made available to callers.
        self.selected_thresholds = apply_best_decoder_thresholds(
            cfg,
            self.threshold_results_path,
        )

    def decode(
        self,
        active_prob: np.ndarray,
        onset_prob: np.ndarray,
        offset_prob: np.ndarray,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        return _decode_probabilities_with_config(
            active_prob,
            onset_prob,
            offset_prob,
            self.cfg.decoder,
        )


def _full_inference_starts(
    total_frames: int,
    chunk_frames: int,
    hop_frames: int,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    if not 0 < hop_frames <= chunk_frames:
        raise ValueError(
            f"hop_frames must be in [1, chunk_frames], got {hop_frames}"
        )
    if total_frames <= chunk_frames:
        return [0]

    final_start = total_frames - chunk_frames
    starts = list(range(0, final_start + 1, hop_frames))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _overlap_blend_weights(
    chunk_frames: int,
    mode: Literal["uniform", "triangular"] = "triangular",
) -> np.ndarray:
    if mode == "uniform":
        return np.ones(chunk_frames, dtype=np.float32)
    if mode != "triangular":
        raise ValueError(f"Unknown overlap blend mode: {mode}")
    position = np.arange(chunk_frames)
    edge_distance = np.minimum(position + 1, chunk_frames - position)
    return (edge_distance / edge_distance.max()).astype(np.float32)


@torch.inference_mode()
def predict_full_feature_matrix(
    model,
    features: np.ndarray,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    hop_frames: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    blend_mode: Literal["uniform", "triangular"] = "triangular",
    device: torch.device | str = "cpu",
    n_pitches: int = CFG.feature.n_pitches,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"Expected features with shape (frames, bins), got {features.shape}"
        )
    if len(features) == 0:
        raise ValueError("Cannot transcribe an empty feature matrix.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")

    model.eval()
    hop_frames = chunk_frames // 2 if hop_frames is None else hop_frames
    total_frames = len(features)
    starts = _full_inference_starts(total_frames, chunk_frames, hop_frames)
    blend = _overlap_blend_weights(chunk_frames, blend_mode)

    probability_sum = np.zeros(
        (total_frames, n_pitches, N_LABEL_CHANNELS),
        dtype=np.float32,
    )
    weight_sum = np.zeros((total_frames, 1, 1), dtype=np.float32)

    for batch_index in tqdm(
        range(0, len(starts), inference_batch_size),
        desc="transcribe",
    ):
        batch_starts = starts[batch_index : batch_index + inference_batch_size]
        batch = np.zeros(
            (len(batch_starts), chunk_frames, features.shape[1]),
            dtype=np.float32,
        )
        valid_lengths = []
        for sample, start in enumerate(batch_starts):
            valid = min(chunk_frames, total_frames - start)
            batch[sample, :valid] = features[start : start + valid]
            valid_lengths.append(valid)

        inputs = torch.from_numpy(batch).unsqueeze(1).to(device, non_blocking=True)
        logits = model(inputs)
        expected_shape = (
            len(batch_starts),
            chunk_frames,
            n_pitches,
            N_LABEL_CHANNELS,
        )
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"Model returned {tuple(logits.shape)}; expected {expected_shape}."
            )
        batch_probabilities = torch.sigmoid(logits).cpu().numpy()

        for sample, (start, valid) in enumerate(zip(batch_starts, valid_lengths)):
            end = start + valid
            local_weight = blend[:valid, None, None]
            probability_sum[start:end] += (
                batch_probabilities[sample, :valid] * local_weight
            )
            weight_sum[start:end] += local_weight

    if np.any(weight_sum == 0):
        raise RuntimeError("Internal chunk-planning error left frames uncovered.")

    probabilities = probability_sum / weight_sum
    return (
        probabilities[..., CH_ACTIVE],
        probabilities[..., CH_ONSET],
        probabilities[..., CH_OFFSET],
    )


def transcribe_feature_matrix(
    model,
    features: np.ndarray,
    decoder: PianoDecoder,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    hop_frames: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    blend_mode: Literal["uniform", "triangular"] = "triangular",
    device: torch.device | str = "cpu",
) -> dict:
    active_prob, onset_prob, offset_prob = predict_full_feature_matrix(
        model,
        features,
        chunk_frames=chunk_frames,
        hop_frames=hop_frames,
        inference_batch_size=inference_batch_size,
        blend_mode=blend_mode,
        device=device,
        n_pitches=decoder.cfg.feature.n_pitches,
    )
    decoded = decoder.decode(active_prob, onset_prob, offset_prob)
    return {
        "active_prob": active_prob,
        "onset_prob": onset_prob,
        "offset_prob": offset_prob,
        **decoded,
    }


def _paired_note_spans(
    onset: np.ndarray,
    offset: np.ndarray,
) -> list[tuple[int, int, int]]:
    spans = []
    total_frames, pitches = onset.shape
    for pitch in range(pitches):
        onset_frames = np.flatnonzero(onset[:, pitch] > 0.5)
        offset_frames = np.flatnonzero(offset[:, pitch] > 0.5)
        offset_index = 0
        for start in onset_frames:
            while (
                offset_index < len(offset_frames)
                and offset_frames[offset_index] <= start
            ):
                offset_index += 1
            end = (
                int(offset_frames[offset_index])
                if offset_index < len(offset_frames)
                else total_frames
            )
            if end > start:
                spans.append((pitch, int(start), end))
            offset_index += 1
    return spans


def decoded_rolls_to_midi(
    onset: np.ndarray,
    offset: np.ndarray,
    output_path: str | Path,
    *,
    feature_cfg: FeatureConfig = CFG.feature,
    default_velocity: float = CFG.decoder.default_velocity,
    tempo: float = 120.0,
    program: int = 0,
) -> Path:
    onset = np.asarray(onset) > 0.5
    offset = np.asarray(offset) > 0.5
    if onset.shape != offset.shape or onset.ndim != 2:
        raise ValueError(
            f"Expected matching (frames, pitches) rolls, got {onset.shape} "
            f"and {offset.shape}."
        )
    if onset.shape[1] != feature_cfg.n_pitches:
        raise ValueError(
            f"Expected {feature_cfg.n_pitches} pitches, got {onset.shape[1]}."
        )
    if not 0 < default_velocity <= 1:
        raise ValueError("default_velocity must be normalized to (0, 1].")

    velocity = int(np.clip(round(default_velocity * 127), 1, 127))
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(
        program=program,
        name="Piano transcription",
    )
    for pitch_index, start, end in _paired_note_spans(onset, offset):
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch_index + feature_cfg.midi_low,
                start=float(start) / feature_cfg.frames_per_second,
                end=float(end) / feature_cfg.frames_per_second,
            )
        )
    instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    midi.instruments.append(instrument)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def transcribe_audio_to_midi(
    model,
    audio_path: str | Path,
    output_midi_path: str | Path,
    *,
    decoder: PianoDecoder | None = None,
    cfg: ExperimentConfig = CFG,
    threshold_results_path: str | Path | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    hop_frames: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    blend_mode: Literal["uniform", "triangular"] = "triangular",
    device: torch.device | str = "cpu",
    program: int = 0,
    return_result: bool = False,
    concert_pitch_hz: float | Literal["auto"] = "auto",
) -> Path | tuple[Path, dict]:
    decoder = decoder or PianoDecoder(cfg, threshold_results_path)
    features = compute_features(audio_path, cfg.feature, concert_pitch_hz=concert_pitch_hz)
    transcription = transcribe_feature_matrix(
        model,
        features,
        decoder,
        chunk_frames=chunk_frames,
        hop_frames=hop_frames,
        inference_batch_size=inference_batch_size,
        blend_mode=blend_mode,
        device=device,
    )
    output_path = decoded_rolls_to_midi(
        transcription["onset"],
        transcription["decoded_offset"],
        output_midi_path,
        feature_cfg=cfg.feature,
        default_velocity=cfg.decoder.default_velocity,
        program=program,
    )
    if return_result:
        return output_path, transcription
    return output_path


def get_sustain_intervals(
    midi: pretty_midi.PrettyMIDI,
    final_time: float,
    sustain_control: int = 64,
    sustain_threshold: int = 64,
) -> list[tuple[float, float]]:
    changes = sorted(
        (
            (float(change.time), int(change.value))
            for instrument in midi.instruments
            if not instrument.is_drum
            for change in instrument.control_changes
            if change.number == sustain_control
        ),
        key=lambda item: item[0],
    )
    intervals: list[tuple[float, float]] = []
    down_time: float | None = None
    for time, value in changes:
        pedal_down = value >= sustain_threshold
        if pedal_down and down_time is None:
            down_time = time
        elif not pedal_down and down_time is not None:
            if time > down_time:
                intervals.append((down_time, time))
            down_time = None
    if down_time is not None and final_time > down_time:
        intervals.append((down_time, final_time))
    return intervals


def sustain_release_for_note_end(
    note_end: float,
    intervals: list[tuple[float, float]],
    interval_starts: list[float],
) -> float:
    interval_index = bisect_right(interval_starts, note_end) - 1
    if interval_index >= 0:
        pedal_down, pedal_up = intervals[interval_index]
        if pedal_down <= note_end < pedal_up:
            return pedal_up
    return note_end


def create_targets(
    midi_path: str | Path,
    n_frames: int,
    *,
    feature_cfg: FeatureConfig = CFG.feature,
) -> np.ndarray:
    """Create pedal-aware active, onset, and offset truth rolls."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    final_time = max(midi.get_end_time(), n_frames / feature_cfg.frames_per_second)
    sustain_intervals = get_sustain_intervals(midi, final_time)
    interval_starts = [start for start, _ in sustain_intervals]
    notes_by_pitch: dict[int, list[pretty_midi.Note]] = {
        pitch: [] for pitch in range(feature_cfg.midi_low, feature_cfg.midi_high + 1)
    }
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if feature_cfg.midi_low <= note.pitch <= feature_cfg.midi_high:
                notes_by_pitch[note.pitch].append(note)

    targets = np.zeros(
        (n_frames, feature_cfg.n_pitches, N_LABEL_CHANNELS),
        dtype=np.uint8,
    )
    for midi_pitch, notes in notes_by_pitch.items():
        notes.sort(key=lambda note: (note.start, note.end))
        pitch_index = midi_pitch - feature_cfg.midi_low
        for note_index, note in enumerate(notes):
            extended_end = sustain_release_for_note_end(
                note.end,
                sustain_intervals,
                interval_starts,
            )
            if note_index + 1 < len(notes):
                effective_end = min(extended_end, notes[note_index + 1].start)
            else:
                effective_end = extended_end

            onset_frame = int(round(note.start * feature_cfg.frames_per_second))
            offset_frame = int(round(effective_end * feature_cfg.frames_per_second))
            if onset_frame < 0 or onset_frame >= n_frames:
                continue
            if offset_frame <= onset_frame:
                offset_frame = onset_frame + 1
            active_end = min(offset_frame, n_frames)
            targets[onset_frame:active_end, pitch_index, CH_ACTIVE] = 1
            targets[onset_frame, pitch_index, CH_ONSET] = 1
            if offset_frame < n_frames:
                targets[offset_frame, pitch_index, CH_OFFSET] = 1
    return targets


def full_piece_note_metrics(
    pred_onset: np.ndarray,
    pred_offset: np.ndarray,
    true_onset: np.ndarray,
    true_offset: np.ndarray,
    onset_tolerance: int,
    offset_tolerance: int,
) -> dict[str, dict[str, float | int]]:
    """Greedy 1:1 note matching with onset and onset+offset scores."""
    predicted_spans = _paired_note_spans(pred_onset, pred_offset)
    true_spans = _paired_note_spans(true_onset, true_offset)

    true_by_pitch: dict[int, list[tuple[int, int]]] = {}
    for pitch, start, end in true_spans:
        true_by_pitch.setdefault(pitch, []).append((start, end))
    used = {
        pitch: np.zeros(len(spans), dtype=bool)
        for pitch, spans in true_by_pitch.items()
    }

    onset_tp = onset_offset_tp = 0
    for pitch, predicted_start, predicted_end in predicted_spans:
        candidates = true_by_pitch.get(pitch, [])
        mask = used.get(pitch)
        best_index = None
        best_distance = None
        for index, (true_start, _) in enumerate(candidates):
            if mask[index]:
                continue
            distance = abs(true_start - predicted_start)
            if (
                distance <= onset_tolerance
                and (best_distance is None or distance < best_distance)
            ):
                best_index = index
                best_distance = distance
        if best_index is None:
            continue
        mask[best_index] = True
        onset_tp += 1
        _, true_end = candidates[best_index]
        if abs(predicted_end - true_end) <= offset_tolerance:
            onset_offset_tp += 1

    predicted_count = len(predicted_spans)
    true_count = len(true_spans)
    return {
        "onset": counts_to_metrics(
            onset_tp,
            predicted_count - onset_tp,
            true_count - onset_tp,
        ),
        "onset_offset": counts_to_metrics(
            onset_offset_tp,
            predicted_count - onset_offset_tp,
            true_count - onset_offset_tp,
        ),
    }


def evaluate_transcription(
    transcription: dict,
    truth: np.ndarray,
    decoder_cfg: DecoderConfig,
) -> dict[str, dict]:
    active_counts = binary_counts(
        transcription["active"] > 0.5,
        truth[..., CH_ACTIVE] > 0.5,
    )
    onset_counts = onset_event_counts(
        transcription["onset"] > 0.5,
        truth[..., CH_ONSET] > 0.5,
        decoder_cfg.onset_tolerance_frames,
    )
    offset_counts = onset_event_counts(
        transcription["decoded_offset"] > 0.5,
        truth[..., CH_OFFSET] > 0.5,
        decoder_cfg.offset_tolerance_frames,
    )
    note_metrics = full_piece_note_metrics(
        transcription["onset"],
        transcription["decoded_offset"],
        truth[..., CH_ONSET],
        truth[..., CH_OFFSET],
        decoder_cfg.onset_tolerance_frames,
        decoder_cfg.offset_tolerance_frames,
    )
    return {
        "frame_active": counts_to_metrics(*active_counts),
        "event_onset": counts_to_metrics(*onset_counts),
        "event_offset": counts_to_metrics(*offset_counts),
        "note": note_metrics,
    }

# Render a MIDI file to WAV using FluidSynth. The MIDI is rendered with the given sample rate and sound font.
def render_midi_to_wav(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int,
    sound_font: str | Path = CFG.decoder.sound_font_path,
) -> Path:
    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    sound_font_path = None if sound_font is None else Path(sound_font)
    if sound_font_path is not None and not sound_font_path.is_file():
        raise FileNotFoundError(f"SoundFont file not found: {sound_font_path}")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("fluidsynth")
    if executable is None:
        raise RuntimeError(
            "FluidSynth executable was not found on PATH. Install FluidSynth "
            "and restart the terminal before running MIDI evaluation."
        )

    command = [
        executable,
        "-ni",
        "-F",
        str(wav_path.resolve()),
        "-r",
        str(sample_rate),
    ]
    if sound_font_path is not None:
        command.append(str(sound_font_path.resolve()))
    command.append(str(midi_path.resolve()))

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(
            f"FluidSynth failed to render {midi_path}.{suffix}"
        ) from error

    if not wav_path.is_file():
        details = (completed.stderr or completed.stdout or "").strip()
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(
            f"FluidSynth did not create the expected WAV: {wav_path}.{suffix}"
        )
    return wav_path

# Render MIDI, transcribe the rendered audio, and evaluate against the original MIDI.
# Perhaps we could use this later in self supervised training
def evaluate_midi_file(
    model,
    midi_path: str | Path,
    *,
    rendered_wav_path: str | Path,
    predicted_midi_path: str | Path,
    metrics_path: str | Path | None = None,
    matrices_path: str | Path | None = None,
    decoder: PianoDecoder | None = None,
    cfg: ExperimentConfig = CFG,
    threshold_results_path: str | Path | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    hop_frames: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    blend_mode: Literal["uniform", "triangular"] = "triangular",
    device: torch.device | str = "cpu",
    concert_pitch_hz: float | Literal["auto"] = "auto"
) -> dict:

    decoder = decoder or PianoDecoder(cfg, threshold_results_path)
    rendered_wav_path = render_midi_to_wav(
        midi_path,
        rendered_wav_path,
        sample_rate=cfg.feature.sample_rate,
        sound_font=cfg.decoder.sound_font_path,
    )
    predicted_midi_path, transcription = transcribe_audio_to_midi(
        model,
        rendered_wav_path,
        predicted_midi_path,
        decoder=decoder,
        cfg=cfg,
        chunk_frames=chunk_frames,
        hop_frames=hop_frames,
        inference_batch_size=inference_batch_size,
        blend_mode=blend_mode,
        device=device,
        return_result=True,
        concert_pitch_hz=concert_pitch_hz
    )
    truth = create_targets(
        midi_path,
        n_frames=len(transcription["active_prob"]),
        feature_cfg=cfg.feature,
    )
    metrics = evaluate_transcription(transcription, truth, cfg.decoder)
    result = {
        "source_midi": str(Path(midi_path)),
        "rendered_wav": str(rendered_wav_path),
        "predicted_midi": str(predicted_midi_path),
        "frames": int(len(truth)),
        "selected_thresholds": decoder.selected_thresholds,
        "metrics": metrics,
    }
    # Don't wanna save matrices rn
    """
    if matrices_path is not None:
        matrices_path = Path(matrices_path)
        matrices_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_probabilities = np.stack(
            [
                transcription["active_prob"],
                transcription["onset_prob"],
                transcription["offset_prob"],
            ],
            axis=-1,
        )
        prediction_decoded = np.stack(
            [
                transcription["active"],
                transcription["onset"],
                transcription["decoded_offset"],
            ],
            axis=-1,
        ).astype(np.uint8)
        np.savez_compressed(
            matrices_path,
            prediction_probabilities=prediction_probabilities,
            prediction_decoded=prediction_decoded,
            truth=truth,
            label_channels=np.array(["active", "onset", "offset"]),
            midi_pitches=np.arange(
                cfg.feature.midi_low,
                cfg.feature.midi_high + 1,
                dtype=np.int16,
            ),
            frames_per_second=np.float32(cfg.feature.frames_per_second),
        )
    """

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["metrics_json"] = str(metrics_path)
    return result