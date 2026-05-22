from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from music21 import converter, midi

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "output"


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() and not path.exists():
        path = DEFAULT_OUTPUT_DIR / path
    return path


def main() -> None:
    parser = ArgumentParser(description="Resonance MIDI player")
    parser.add_argument("midi_file", help="MIDI filename or full path")
    args = parser.parse_args()

    path = _resolve_path(args.midi_file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Playing: {path.name}")
    score = converter.parse(str(path))
    sp = midi.realtime.StreamPlayer(score)
    sp.play()


if __name__ == "__main__":
    main()