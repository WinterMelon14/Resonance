from __future__ import annotations

from pathlib import Path
from typing import Any
from midiutil import MIDIFile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(file_path: Path | str, sr: int | None = None) -> dict[str, Any]:
    """Load an audio file and return the signal metadata."""
    path = Path(file_path)
    signal, sample_rate = librosa.load(str(path), sr=sr, mono=False)
    print("Sample rate is", sample_rate)
    channels = 1 if signal.ndim == 1 else signal.shape[0]
    duration = float(signal.shape[-1]) / sample_rate
    return {
        "signal": signal,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "path": path,
    }


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    """Normalize audio so the peak absolute value is 1.0."""
    peak = float(np.max(np.abs(signal)))
    if peak <= 0:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert a multi-channel signal to mono."""
    if signal.ndim == 1:
        return signal
    return np.mean(signal, axis=0).astype(signal.dtype)


def audio_info(signal: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    """Return metadata about the loaded audio."""
    samples = signal.shape[-1]
    duration = float(samples) / sample_rate
    peak = float(np.max(np.abs(signal))) if samples else 0.0
    rms = float(np.sqrt(np.mean(signal**2))) if samples else 0.0
    return {
        "sample_rate": sample_rate,
        "duration": duration,
        "samples": samples,
        "peak": peak,
        "rms": rms,
    }


def save_waveform(signal: np.ndarray, sample_rate: int, output_path: Path) -> None:
    """Save a waveform plot to an image file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(signal.shape[-1]) / sample_rate   
    plt.figure(figsize=(10, 3))
    plt.plot(times, signal, color="#007acc", linewidth=0.6)
    plt.fill_between(times, signal, color="#007acc", alpha=0.2)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def save_spectrogram(signal: np.ndarray, sample_rate: int, output_path: Path) -> None:
    """Save a spectrogram image for a mono audio signal."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if signal.ndim != 1:
        signal = to_mono(signal)
    # In save_spectrogram, replace the stft call with:
    stft = np.abs(librosa.stft(signal, n_fft=2048, hop_length=256))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        db,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

def compute_fft_piano(
    signal: np.ndarray,
    sample_rate: int,
    freq_range: tuple[float, float] = (27.5, 4186.0),  # A0 to C8
) -> dict[str, np.ndarray]:
    """FFT scoped to the actual piano frequency range (A0–C8)."""
    mono = to_mono(signal)
    n = len(mono)
    fft_result = np.fft.rfft(mono * np.hanning(n))  # Hanning window reduces spectral leakage
    magnitudes = np.abs(fft_result)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    db = librosa.amplitude_to_db(magnitudes, ref=np.max)

    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    return {
        "frequencies": frequencies[mask],
        "magnitudes": magnitudes[mask],
        "db": db[mask],
    }




def estimate_note_pitches(
    signal: np.ndarray,
    sample_rate: int,
    onset_times: np.ndarray,
    frame_duration: float = 0.25,  # was 0.08 — gives pyin more cycles to lock on
) -> list[dict]:
    """Estimate the dominant pitch (MIDI + note name) at each onset."""
    mono = to_mono(signal)
    notes = []
    for t in onset_times:
        attack_offset = 0.03  # 30 ms

        start = int((t + attack_offset) * sample_rate) # delay it to skip heavy hammer strike
        end = int((t + frame_duration) * sample_rate)
        frame = mono[start:end]
        if len(frame) < 512:
            continue
        f0, voiced, _ = librosa.pyin(
            frame,
            fmin=27.5,   # A0
            fmax=4186.0, # C8
            sr=sample_rate,
            frame_length=4096
        )
        voiced_f0 = f0[voiced] if f0 is not None else np.array([])
        if len(voiced_f0) == 0:
            continue
        freq = float(np.median(voiced_f0))
        midi = int(round(librosa.hz_to_midi(freq)))
        notes.append({
            "time": round(float(t), 4),
            "freq_hz": round(freq, 2),
            "midi": midi,
            "note": librosa.midi_to_note(midi),
        })
    return notes

def detect_onsets_piano(
    signal: np.ndarray,
    sample_rate: int,
    units: str = "time",
    delta: float = 0.3,
    hop_length: int = 512,
) -> np.ndarray:
    mono = to_mono(signal)

    onset_env = librosa.onset.onset_strength(
        y=mono,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=4096,
        aggregate=np.median,   # median instead of mean — ignores amplitude flutter
        center=True,
    )

    # Smooth the envelope to kill the intra-note amplitude modulation peaks
    onset_env = np.convolve(onset_env, np.hanning(6), mode="same")
    onset_env /= onset_env.max()  # renormalise after smoothing

    # At 126 BPM a 16th note = ~119ms
    # wait = 119ms / (hop_length / sample_rate) = 119ms / 11.6ms ≈ 10 frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        pre_max=4,
        post_max=4,
        pre_avg=12,
        post_avg=12,
        delta=delta,
        wait=10,
        backtrack=True,
        units=units,
    )
    return onset_frames

def save_onset_plot(
    signal: np.ndarray,
    sample_rate: int,
    onset_times: np.ndarray,  # pre-computed, passed in
    output_path: Path,
) -> None:
    """Save a waveform plot with onset markers overlaid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(len(signal)) / sample_rate

    plt.figure(figsize=(10, 3))
    plt.plot(times, signal, color="#007acc", linewidth=0.6)
    plt.vlines(onset_times, -1, 1, color="#e05c00", linewidth=0.9, alpha=0.8, label="Onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform with Onsets ({len(onset_times)} detected)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

def notes_to_midi(
    notes: list[dict],
    output_path: Path,
    tempo: int = 120,
    track_name: str = "Piano",
) -> None:
    """Convert detected notes to a MIDI file.
    
    Each note's duration is inferred from the gap to the next onset;
    the last note gets a fixed fallback duration.
    """
    if not notes:
        raise ValueError("No notes to write.")

    midi = MIDIFile(numTracks=1)
    midi.addTempo(track=0, time=0, tempo=tempo)
    midi.addTrackName(track=0, time=0, trackName=track_name)

    beats_per_second = tempo / 60.0

    for i, note in enumerate(notes):
        # Duration = gap to next onset, capped at 2s; last note gets 0.5s
        if i < len(notes) - 1:
            gap = notes[i + 1]["time"] - note["time"]
            duration_secs = min(gap, 2.0)
        else:
            duration_secs = 0.5

        onset_beats = note["time"] * beats_per_second
        duration_beats = duration_secs * beats_per_second

        midi.addNote(
            track=0,
            channel=0,
            pitch=note["midi"],
            time=onset_beats,
            duration=duration_beats,
            volume=100,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        midi.writeFile(f)


def save_fft_plot(fft_data: dict[str, np.ndarray], output_path: Path) -> None:
    """Save an FFT frequency spectrum plot from pre-computed FFT data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(fft_data["frequencies"], fft_data["db"], color="#e05c00", linewidth=0.8)
    plt.fill_between(fft_data["frequencies"], fft_data["db"], fft_data["db"].min(), color="#e05c00", alpha=0.15)
    plt.xscale("log")
    plt.xlim(fft_data["frequencies"][0], fft_data["frequencies"][-1])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT Spectrum")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
