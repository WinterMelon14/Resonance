---
description: Describe when these instructions should be loaded by the agent based on task context
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

This is my project plan for a music transcription software. The software will have two main features: converting music files to audio and converting audio to music files.

I. Music file --> Audio player: Subfeatures: MIDI (easy), PDF (hard), User camera image upload (Very hard, requires AI and thus a backend, unless the model is on the frontend but that sounds disgusting), MSCZ (optional) II. Audio --> Music file: Subfeatures: ML model to detect what notes are being played. This will be especially difficult for chords. I imagine we can do some detective work with waves as sound is waves converted to analog signals, and we can use the signals in the file combined with beat frequency/other waves related physics formulas on top of our machine learning model. Subfeature 2: Learning/Finding WHEN notes are played Subfeature 3: Deducing Time Signature. Toby Fox's music makes this very challenging. Subfeature 4: Using volume to add dynamic notations Subfeature 5: Any additional "cherry on tops," like articulations, cresc/dims, rits, etc.

It will look something like this:
Resonance/
│
├── README.md
├── docker-compose.yml
├── .gitignore
├── scripts/
│   ├── setup.sh
│   ├── dev.sh
│   └── test_sample.sh
│
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   │
│   ├── public/
│   └── src/
│       ├── app/
│       ├── components/
│       ├── hooks/
│       ├── lib/
│       │   ├── api.ts
│       │   └── tone.ts
│       └── types/
│
├── backend/
│   ├── requirements.txt
│   ├── pyproject.toml
│   │
│   ├── src/
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   │   ├── transcription.py
│   │   │   │   └── upload.py
│   │   │   └── schemas/
│   │   │
│   │   ├── cli/
│   │   │   └── main.py
│   │   │
│   │   ├── core/
│   │   │   ├── audio/
│   │   │   │   └── dsp/
│   │   │   ├── transcription/
│   │   │   ├── formats/
│   │   │   └── ml/
│   │   │
│   │   ├── services/
│   │   │   ├── pdf_to_midi.py
│   │   │   ├── image_to_midi.py
│   │   │   └── audio_to_midi.py
│   │   │
│   │   └── utils/
│   │
│   ├── tests/
│   └── data/
│       ├── samples/
│       └── output/
│
└── docs/
    └── architecture.md
(Let's not worry about the frontend for now, or creating all these files yet.)

For now, I want to establish a foundation and we are at phase 1.

Phase 1 — Audio Loading + Visualization

Goal: touch real audio ASAP.

Audio Loading
 Load WAV files
 Load MP3 files
 Normalize audio
 Convert stereo → mono
CLI
 Create CLI entrypoint
 Add:
 info
 waveform
 spectrogram

Example:python -m cli.main spectrogram song.wav

Visualization
 Plot waveform
 Generate FFT
 Generate spectrogram
 Save spectrogram image
