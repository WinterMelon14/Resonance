from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BACKEND_DIR = Path(__file__).resolve().parent

@dataclass
class PathConfig:
    work_dir: Path = BACKEND_DIR / "data" / "output"
    checkpoint_path: Path = (
        BACKEND_DIR
        / "data"
        / "models"
        / "piano_transcriber_test_gru2_pedal_epoch_20.pt"
    )


@dataclass
class FeatureConfig:
    kind: Literal["cqt", "log_mel"] = "cqt"
    sample_rate: int = 44_100
    hop_length: int = 384
    cqt_bins_per_semitone: int = 3
    cqt_bins_per_octave: int = 36
    cqt_extract_bins: int = 344
    midi_low: int = 21
    midi_high: int = 108

    @property
    def n_pitches(self) -> int:
        return self.midi_high - self.midi_low + 1

    @property
    def n_bins(self) -> int:
        return self.n_pitches * self.cqt_bins_per_semitone

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass
class ModelConfig:
    name: Literal["baseline"] = "baseline"
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.10


@dataclass
class DecoderConfig:
    active_start_threshold: float = 0.60
    active_continue_threshold: float = 0.40
    active_end_threshold: float = 0.10
    active_end_patience_frames: int = 3
    onset_threshold: float = 0.25
    onset_tolerance_frames: int = 2
    active_support_threshold: float = 0.25
    active_support_radius: int = 2
    onset_pre_max: int = 3
    onset_post_max: int = 3
    onset_min_gap_frames: int = 4
    offset_threshold: float = 0.25
    offset_pre_max: int = 3
    offset_post_max: int = 3
    offset_min_gap_frames: int = 4
    offset_tolerance_frames: int = 6
    min_note_frames: int = 2
    use_offset_head: bool = True
    offset_active_drop: float = 0.20
    offset_lookahead_frames: int = 4
    offset_peak_window_frames: int = 12
    default_velocity: float = 0.80
    sound_font_path: Path = Path("./backend/data/soundfonts/musescore.sf3")


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


CFG = ExperimentConfig()