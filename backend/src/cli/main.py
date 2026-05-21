from pathlib import Path
from music21 import converter, midi

BASE_DIR = Path(__file__).parent.parent.parent
midi_path = BASE_DIR / 'data' / 'guardian.mid'

score = converter.parse(str(midi_path))
sp = midi.realtime.StreamPlayer(score)
sp.play()