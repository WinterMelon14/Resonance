"""Run Resonance's evaluate-midi command sequentially for every MIDI file.

Put this file in ``backend/`` and run:

    python batch_evaluate_midi.py

The active virtual environment's Python interpreter is reused. No third-party
packages are required.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MIDI_DIR = BACKEND_DIR / "data" / "audio"

# True includes MIDI files in subfolders; False scans MIDI_DIR itself only.
RECURSIVE = False

# When True, one failed file is reported and the remaining files are attempted.
CONTINUE_ON_ERROR = True


def find_midi_files() -> list[Path]:
    iterator = MIDI_DIR.rglob("*") if RECURSIVE else MIDI_DIR.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )


def main() -> int:
    if not MIDI_DIR.is_dir():
        raise FileNotFoundError(f"MIDI folder does not exist: {MIDI_DIR}")

    midi_files = find_midi_files()
    if not midi_files:
        print(f"No .mid or .midi files found in {MIDI_DIR}")
        return 0

    print(f"Found {len(midi_files)} MIDI files in {MIDI_DIR}")
    print(f"Using Python: {sys.executable}\n")

    failures: list[tuple[Path, int]] = []
    batch_start = time.perf_counter()

    for index, midi_path in enumerate(midi_files, start=1):
        print(f"[{index}/{len(midi_files)}] {midi_path.name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "backend.transcribe",
            "evaluate-midi",
            str(midi_path),
        ]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        elapsed = time.perf_counter() - started

        if result.returncode == 0:
            print(f"Completed in {elapsed:.1f}s\n")
        else:
            failures.append((midi_path, result.returncode))
            print(f"FAILED with exit code {result.returncode} after {elapsed:.1f}s\n")
            if not CONTINUE_ON_ERROR:
                break

    total_elapsed = time.perf_counter() - batch_start
    completed = len(midi_files) - len(failures)
    print(f"Finished: {completed}/{len(midi_files)} succeeded in {total_elapsed:.1f}s")

    if failures:
        print("\nFailures:")
        for path, return_code in failures:
            print(f"  {path.name} (exit code {return_code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())