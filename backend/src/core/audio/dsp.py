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
