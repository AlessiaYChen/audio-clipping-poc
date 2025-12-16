# ðŸ“¡ News Audio Story Segmentation

### Hybrid Audio + Text Pipeline for Radio Story Boundary Detection

This project performs **automatic segmentation of radio newscasts** (CKNW, CFAX, CBC, CHNL, CKFR) into **story-level audio clips** using a hybrid approach that combines:

* **Audio features** (VAD, diarization, audio embeddings, acoustic change detection)
* **Transcription + text features** (word timestamps, semantic embeddings, topic shifts)
* **Heuristic + data-driven boundary detection**
* **FFmpeg-based clipping**
* **Evaluation harness** with precision/recall/F1 metrics and parameter sweeps

> **Goal:** Production-quality, robust, station-independent story segmentation for continuous 24/7 radio monitoring.

---

## ðŸš€ Features

* ðŸŽ§ **Audio track analysis**

  * Voice Activity Detection (VAD)
  * Speaker diarization (anchor detection, studio/field separation)
  * Short-window audio embeddings (3â€“5 s)
  * Acoustic change-point detection
  * Silence & jingle detection

* ðŸ“ **Text track analysis**

  * Whisper/Azure STT transcription
  * Word-aligned timestamps
  * Text chunking (5â€“10 s)
  * Semantic embeddings
  * Topic shift detection

* ðŸ” **Hybrid boundary detection**

  * Combines audio + text + structural heuristics
  * Scored boundary candidates with reasons
  * Station-specific tunable configs

* âœ‚ï¸ **Clipper**

  * Clean, gap-free, timestamp-accurate FFmpeg clipping

* ðŸ“Š **Evaluation harness**

  * GT boundary loading
  * Tolerance-based matching
  * Precision/recall/F1
  * Per-station metrics
  * Threshold sweeps for scientific tuning

---

# ðŸ“¦ Architecture Overview

```
audio file
   â”‚
   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚    Ingestor      â”‚  â†’ normalize + resample (16k mono WAV)
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
   â”‚
   â–¼

        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚         Dual Feature Extraction        â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚      Audio Track          â”‚      â”‚          Text Track          â”‚
  â”‚  - VAD                    â”‚      â”‚  - Transcription (Whisper)   â”‚
  â”‚  - Diarization            â”‚      â”‚  - Word timestamps           â”‚
  â”‚  - Audio embeddings       â”‚      â”‚  - Text chunking (5â€“10 s)    â”‚
  â”‚  - Jingle/silence detect  â”‚      â”‚  - Semantic embeddings       â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â”‚  - Semantic shift detection  â”‚
                                     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
   â”‚                                           â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â–¼
            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
            â”‚        Hybrid Boundary Detector        â”‚
            â”‚  - score candidates                    â”‚
            â”‚  - merge audio + text signals          â”‚
            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
                          â–¼
            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
            â”‚             Segment Planner            â”‚
            â”‚ - prune, dedupe, min length            â”‚
            â”‚ - snap to word boundary / energy dip   â”‚
            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
                          â–¼
            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
            â”‚                Clipper                 â”‚
            â”‚     - FFmpeg cuts per segment          â”‚
            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
                          â–¼
               Output:
               - segments.json
               - transcript.json
               - segments/*.wav

```

A high-resolution PDF version is available (`docs/news_audio_architecture.pdf`).

---

# ðŸ“ Repository Structure

Recommended layout:

```
codex-audio/
  README.md
  pyproject.toml
  codex_audio/
    __init__.py
    config.py
    models.py
    cli.py

    ingest.py
    io_utils.py

    transcription/
      whisper_backend.py

    features/
      vad.py
      diarization.py
      embeddings.py
      lowlevel.py

    text_features/
      segments.py
      embeddings.py
      change_points.py

    segmentation/
      candidates.py
      scoring.py
      planner.py

    clipping/
      ffmpeg_wrapper.py

    evaluation/
      io.py
      matching.py
      metrics.py
      sweep.py
      cli.py

  docs/
    news_audio_architecture.pdf
    examples.md
  tests/
    test_ingest.py
    test_features.py
    test_segmentation.py
    test_evaluation.py
```

---

# ðŸ›  Installation

### Requirements:

* Python 3.10+
* FFmpeg installed on system
* Whisper or Azure STT
* Pyannote (optional but recommended for diarization)

```bash
pip install -r requirements.txt
```

---

# ðŸ§ª Running the Pipeline

## Azure Transcription & Diarization

1. Set `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` in your shell (or `.env`).
2. Add a `transcription` block to your station config so Azure STT can emit speaker IDs:

```yaml
transcription:
  provider: azure_speech
  diarization: true
  max_speakers: 6
```

3. Run the CLI normally:

```bash
codex-audio segment examples/CBU_730z_251119_072800.mov \
  --station CKNW \
  --config config/stations/CKNW.yaml \
  --out out/diarized
```

With `diarization: true`, the emitted `transcript.json` includes a `speaker` field for every word, and the pipeline still performs the LLM quote alignment and silence refinement automatically.

### Segment an audio file

```bash
codex-audio segment input.wav \
  --station CKNW \
  --out out_dir/
```

Outputs:

```
out_dir/
  segments/
    segment_000.wav
    segment_001.wav
  segments.json
  transcript.json
```

---

# ðŸ“Š Evaluation Harness

### Manifest format (`evaluation_manifest.csv`)

```csv
audio_path,annotation_path,station
data/CKNW_2025-01-15.wav,annotations/CKNW_2025-01-15.csv,CKNW
data/CFAX_2025-01-12.wav,annotations/CFAX_2025-01-12.csv,CFAX
```

### Ground truth annotation CSV

```csv
time_s
61.3
185.9
402.4
...
```

### Run evaluation

```bash
codex-audio eval \
  --manifest evaluation_manifest.csv \
  --tolerance-s 3.0 \
  --config station_config.yaml
```

Output example:

```
Overall:
  Precision = 0.78
  Recall    = 0.74
  F1        = 0.76

CKNW:
  Precision = 0.80
  Recall    = 0.71
  F1        = 0.75
```

---

# ðŸ”¬ Threshold Sweeps

```bash
codex-audio sweep \
  --manifest evaluation_manifest.csv \
  --param silence_min_s 0.7 1.0 1.3 \
  --param min_boundary_score 3.0 4.0 5.0
```

Generates `sweep_results.csv`:

```
silence_min_s,min_boundary_score,F1
1.0,4.0,0.78
0.7,3.0,0.74
...
```

---

# ðŸ§© Extending the Pipeline

Future enhancements:

* LLM-based story labeling (title + summary per segment)
* Automatic jingle classifier per station
* Real-time streaming segmentation
* Station-specific configuration auto-tuning
* Combined audio+video segmentation (future)

---

# ðŸ¤ Contributing

PRs welcome!
Please file an issue before adding major features.
Add tests for all new modules.

---

If you want, I can also:

âœ… Generate the scaffolding project folder with empty modules
âœ… Produce the `pyproject.toml` or `requirements.txt`
âœ… Write the initial CLI implementation
âœ… Create the `examples/` folder with realistic sample outputs

Just tell me!

