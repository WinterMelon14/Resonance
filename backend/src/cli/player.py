from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pygame

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
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()

    # Block until playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)


if __name__ == "__main__":
    main()