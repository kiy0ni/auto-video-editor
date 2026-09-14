# Auto Video Editor

[![Tests](https://github.com/kiy0ni/auto-video-editor/actions/workflows/tests.yml/badge.svg)](https://github.com/kiy0ni/auto-video-editor/actions/workflows/tests.yml)

Turn a long recording (Twitch/YouTube stream, VOD, podcast, gameplay session) into:

- **a highlight reel** (16:9) built from the best moments, with clean cuts that never land mid-sentence,
- **vertical shorts** (9:16, 1080x1920) for TikTok / YouTube Shorts / Reels, with **animated burned-in captions**,
- **hand-off files for your editing software**: EDL and Final Cut Pro 7 XML (Premiere Pro, DaVinci Resolve, Avid), subtitles, YouTube chapters and a CSV of every moment.

Everything runs locally: FFmpeg for audio/video, Whisper for speech recognition.

**Fully automatic by default.** Drop a video and click *Create*: the app looks at the picture, listens to a
few samples and decides the style, the speech model, the reel length, the number of shorts, the vertical
layout (including where your webcam overlay is), the vocabulary of the game you play and where the stream
really starts and ends. Every automatic choice is shown and can be overridden.

![workflow](https://img.shields.io/badge/workflow-analyze%20%E2%86%92%20review%20%E2%86%92%20render-4f8cff)

---

## Features

**Automatic choices** (each one can be set by hand instead)

| Setting | How it is decided |
|---------|-------------------|
| Content style | Game recognised, webcam overlay, speech coverage, background audio, faces and motion → *gaming*, *talk* or *vlog* preset |
| Speech model | NVIDIA GPU → `large-v3`, Apple Silicon / 8+ cores → `small`, otherwise `base` (lighter for very long recordings) |
| Language | Voted on three samples spread over the recording |
| Vocabulary | One game and one topic recognised among 100+ packs from the title, the metadata or a speech sample → their names and jargon are given to the transcriber |
| Hype phrases | Built-in phrases for the detected language, plus laughter detection |
| Reel length | Follows the total length of the strong moments (1 min to 20 min, never more than 12-30% of the source) |
| Number of shorts | One per great moment (up to 3, 6 or 10 depending on the length of the recording) |
| Shorts layout | Webcam overlay found → facecam split with the detected area; face on screen → face tracking; otherwise blur fill |
| Intro / outro | Long parts at the start or the end where nobody talks are skipped |

**Moment detection**
- Single-pass audio analysis (fast even on 10-hour recordings) that scores every second of the recording:
  loudness above the *normal level of the stream* (a shout stands out even in a loud game, but calm talk
  in a quiet intro does not), sudden spikes, speech rate, exclamations and **hype phrases**
  ("no way", "let's go", "clip that", your own list...).
- A quality gate: weak moments are never used to pad the reel. A shorter reel beats a boring one.
  Optionally skip the start and the end of the recording (waiting screen, outro).
- Speech recognition with word-level timestamps (faster-whisper or openai-whisper). Cuts snap to
  sentence boundaries; without a speech engine the cuts snap to the quietest nearby point instead.
- Moments are spread over the whole recording instead of clustering in one section.

**Transcription quality**
- **Vocabulary packs**: 100+ built-in packs of names and jargon, each recognised in titles by its aliases:
  - *Games* (69): Minecraft, Fortnite, Valorant, League of Legends, TFT, CS2, Apex, Overwatch, R6, Warzone,
    GTA RP, Rocket League, EA FC, Elden Ring, Baldur's Gate 3, Zelda, Pokémon, Roblox, Lethal Company,
    Phasmophobia, Dead by Daylight, WoW, Genshin, Chess, GeoGuessr, osu!, Stardew Valley, Terraria...
  - *Sports*: football, NBA, Formula 1, MMA & boxing, tennis, rugby, cycling & running, esports.
  - *Streaming & culture*: Twitch slang (FR), internet slang FR/EN, IRL & vlog.
  - *Talk & podcast*: podcast & interview, true crime, business/finance/crypto, news & society.
  - *Tech & science*: programming, AI, PC hardware, science & space, gadgets, school.
  - *Lifestyle*: cooking, fitness, beauty & fashion, travel, cars, DIY, pets.
  - *Arts & entertainment*: music production, rap & concerts, drawing, movies & series, anime & manga,
    comedy, paranormal, books.

  Pick them in the app (search, categories, preview of every term) or let the app recognise them.
  Only the most relevant terms fit in the model's prompt: your own words first, then the terms heard in
  a speech sample. Add your own packs (your community, server, running jokes) as `.json` files in the
  packs folder.
- **Vocabulary hints**: your own names, jargon and nicknames, always given to the model first.
- **Swear words** are transcribed instead of silently skipped, and can be **censored** (p*tain, sh*t)
  in captions, subtitles and reports, with your own extra words.
- Language is detected on real speech (not on the silent intro), and hallucinations over silence or
  repeated words are filtered out.
- **Speaker detection** (built-in, no extra install): each sentence gets a voice signature and the
  voices are clustered. Captions get one color per speaker, the transcript is labelled *A:* / *B:*,
  and caption lines never mix two speakers.

**Editing quality**
- Frame-accurate cuts, source frame rate preserved, 20 ms audio fades on every cut (no clicks or pops).
- Optional **jump cuts**: dead air is removed inside moments, while loud action without speech is kept.
- Optional fade transitions, lead-in and tail padding.
- **Two-pass loudness normalization** (EBU R128, -14 LUFS by default) for consistent volume.
- Hardware encoding when available (VideoToolbox on macOS, NVENC / Quick Sync / AMF on Windows & Linux),
  with an automatic fallback to x264.

**Shorts**
- Standalone moments of 20-59 s (configurable), starting on a sentence.
- Optional **hook**: the short opens with its best 3 seconds, then plays from the start.
- Layouts: `blur` (full frame over a blurred background), `crop` (center crop),
  `smart` (crop follows the detected face), `split` (facecam on top, gameplay below).
- Karaoke-style captions (active word highlighted) or simple captions, placed away from busy areas.

**Publishing kit**
- A thumbnail for the reel and a cover for every short, picked as the sharpest, well exposed frame around the peak.
- A suggested title (the punchiest sentence), a description and hashtags (game, style, language) for every
  short, and a title + YouTube chapters for the reel.

**Workflow**
- Desktop app with a **review step**: preview each moment, include/exclude it, then render.
- Command line interface for automation and batch processing (`python main.py *.mp4`).
- Editable `project.json`: analyze once, tweak the moment list, render later.
- Analysis and transcripts are cached, so re-running with other settings takes seconds.
- Cancel at any time; partially written files are cleaned up.

---

## Installation

1. **Python 3.9+**
2. **FFmpeg** (with `ffprobe`, and `ffplay` for previews) on your PATH:
   - macOS: `brew install ffmpeg`
   - Windows: `winget install ffmpeg` (or download from https://ffmpeg.org/download.html)
   - Linux: `sudo apt install ffmpeg`
3. Python packages:

```bash
git clone https://github.com/kiy0ni/auto-video-editor.git
cd auto-video-editor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy + faster-whisper + OpenCV
```

`numpy` is the only hard requirement. `faster-whisper` is strongly recommended (sentence-aware cuts,
keywords, captions). OpenCV is only used by the `smart` shorts layout. You can also install it as a
package with `pip install ".[all]"`, which adds an `auto-video-editor` command.

The Whisper model is downloaded automatically the first time it is used.

---

## Usage

### Desktop app

```bash
python main.py
```

The app has two modes, switchable at the top of the sidebar.

**Simple mode** (default), everything happens on one page:

1. **Drop a recording** on the window (or click *Browse files*).
   *What we found* shows what the picture scan detected (webcam overlay, faces, motion, game) and a rough
   processing time for this computer.
2. Pick **what to create**: reel + shorts, reel only or shorts only.
3. Leave the kind of video on **Auto-detect**, or pick *Gaming stream*, *Podcast & talk* or *Vlog & IRL*.
   The style tunes moment detection, pacing (jump cuts, fades) and the shorts layout.
4. Adjust the essentials (reel length, number of shorts, vertical format, captions, language,
   vocabulary hints, swear word censoring; most are on *Auto*) and click **Create my edit**. The *Review*
   page lists the automatic choices that were made. Or **Review first** to preview, keep or drop each moment before rendering.

**Advanced mode** adds sidebar pages that expose every setting of the engine:

- **Highlight reel**: length, shortest/longest moment, build-up and reaction kept around each peak, padding,
  how much the selection spreads over the recording, minimum score, skipped start/end, transition and
  fade duration, jump cuts with pause length and silence level.
- **Shorts & captions**: count and length, visual layout picker, facecam area drawn on a frame of your video,
  background blur, caption style, font, size, position, uppercase, words per line, highlight color and
  color by speaker.
- **Detection**: speech engine, model, language, beam size, vocabulary packs (picker and automatic
  recognition), vocabulary hints, swear words (transcribe /
  censor / extra words), speaker detection and count, the weight of each scoring signal (loudness,
  spikes, speech rate, hype phrases, exclamations) and the hype phrase list.
- **Export**: encoder, quality, loudness target, audio bitrate, SRT/EDL/XML, reset to defaults.

Mode, theme, settings and the last analysis of a video are remembered. Shortcuts: `Cmd/Ctrl+O` open a video,
`Cmd/Ctrl+Enter` render, `Cmd/Ctrl+L` activity log, `Cmd/Ctrl+1…6` pages.

### Command line

```bash
# Everything with the defaults (medium highlight + 5 blurred-background shorts)
python main.py stream.mp4

# 8-minute reel with jump cuts, 8 face-tracked shorts, French speech
python main.py stream.mp4 --target 8m --jump-cuts --shorts 8 --layout smart --language fr

# Facecam in the top-right corner: split layout
python main.py stream.mp4 --layout split --facecam 0.72,0.02,0.26,0.30

# Only a highlight reel, no transcription
python main.py stream.mp4 --shorts 0 --transcriber none

# Two-step workflow: analyze, edit the project file, render
python main.py stream.mp4 --analyze-only
python main.py --project stream_edit/project.json

# French Minecraft stream with a co-streamer: vocabulary hints, censored swear words, skip the intro
python main.py stream.mp4 --language fr --vocabulary "Minecraft,creeper,mob,cave" --censor --speakers 2 --skip-start 3m

# Vocabulary packs (python main.py --list-packs shows all of them)
python main.py stream.mp4 --packs minecraft,twitch-fr

# Presets and fine-tuning of any setting
python main.py podcast.mp4 --preset talk --set caption_size=130 --set caption_color=#39FF88
python main.py --list-settings      # every setting with its default value
```

Run `python main.py --help` for all options.

---

## Output

For `stream.mp4` the output folder `stream_edit/` contains:

| File | Description |
|------|-------------|
| `stream_highlight.mp4` | The highlight reel |
| `stream_highlight.srt` | Subtitles for the reel |
| `stream_highlight.edl` | CMX 3600 EDL of the reel, pointing at the original file |
| `stream_highlight.xml` | Final Cut Pro 7 XML (import in Premiere Pro or DaVinci Resolve) |
| `stream_chapters.txt` | YouTube chapters (`00:00 Title`) for the reel description |
| `stream_moments.csv` | Every moment: source timecodes, position in the reel, score, reasons, transcript |
| `clips/clip_001_12m34s.mp4` | Each moment as a separate file |
| `stream_thumbnail.jpg`, `stream_publish.txt` | Reel thumbnail, title, chapters and hashtags |
| `shorts/short_01_<title>.mp4` + `.srt` | Vertical shorts, best first |
| `shorts/short_01_<title>.jpg` + `.txt` | Cover image, title, description and hashtags of each short |
| `project.json` | The editable plan (see below) |

### Fine-tuning in Premiere Pro / DaVinci Resolve

Import `stream_highlight.xml` (Premiere: *File > Import*; Resolve: *File > Import > Timeline*) to get
the edit as a timeline on the **original** recording, so you can extend a cut, reorder moments or add
graphics without losing quality. Use the `.edl` for Avid or if an XML import fails.

### Editing `project.json`

Each moment has `start` / `end` (seconds in the source), `selected`, `score`, `reasons` and `text`.
Set `"selected": false` to drop a moment, or change `start` / `end`, then render with `--project`.
Captions and jump cuts follow the edited times.

---

## How it works

1. **Probe** the file with ffprobe (duration, frame rate, resolution, timecode, title metadata), then
   **scan the picture**: a few dozen frame grabs measure motion and find faces and a webcam overlay (cached).
2. **Audio envelope**: one ffmpeg pass decodes the audio to 16 kHz mono and computes RMS/peak levels
   every 50 ms (cached).
3. **Speech preview**: three 30 s samples give the language and the game or topic (cached).
   **Transcription** in 10-minute windows with word timestamps (cached), conditioned on the vocabulary
   hints. Words over silent audio and zero-confidence repetitions are discarded as hallucinations.
   Each sentence also gets a voice signature (timbre + pitch) used to tell speakers apart.
4. **Scoring**: per-second score = loudness above max(rolling 5 min median, typical level of the whole
   recording) + spikes + speech rate + hype phrases + exclamations, smoothed.
5. **Candidates**: peaks are grown into regions, given a build-up before and a reaction after, snapped
   to sentence boundaries, clamped to the min/max length and de-overlapped.
6. **Selection**: best moments first with a penalty for clustering, until the target length is reached.
   Shorts are chosen separately and extended or trimmed to 20-59 s.
7. **Render**: each moment is cut with a single ffmpeg filter graph (trim + concat for jump cuts,
   vertical layout, captions, loudness), then the reel is assembled losslessly and normalized.

### Highlight length presets

| Preset | Length |
|--------|--------|
| `short` | ~5% of the source, between 1 and 10 minutes |
| `medium` | ~10% of the source, between 2 and 20 minutes |
| `long` | ~20% of the source, between 4 and 40 minutes |

Never more than half of the source. Use `--target` (or *Exact length*) to set it yourself.

---

## Tips & troubleshooting

- **Speed**: transcription is the slowest step. `small` (the default) is a good balance on CPU; `base`
  is twice as fast but misses more words; with an NVIDIA GPU, `large-v3` or `turbo` is affordable.
  Runs after the first one reuse the cache.
- **Wrong language detected / anglicisms mangled**: set the language explicitly (`--language fr`) and add
  the words to the vocabulary hints.
- **Speakers mixed up**: set the number of speakers instead of auto, or turn detection off. The built-in
  detector relies on voices sounding different; two similar voices on the same microphone will be merged.
- **Moments feel too short/long**: tune *Shortest/Longest moment* and the lead-in/tail padding.
- **Your own catchphrases**: add them in *Analysis > Hype phrases* (or `--keywords`).
- **Hardware encoder errors**: choose `libx264` in *Output > Video encoder*.
- **`smart` layout falls back to a centered crop**: install OpenCV 4 (`pip install "opencv-python-headless<5"`).
  Faces are only detected when clearly visible; for streams with a facecam overlay, `split` usually looks better.
- Caches live in `~/Library/Caches/auto-video-editor` (macOS), `%LOCALAPPDATA%\auto-video-editor` (Windows)
  or `~/.cache/auto-video-editor` (Linux). Delete the folder to reclaim space.

---

## Custom vocabulary packs

Click *My packs folder* in the app (or look at the path printed by `python main.py --list-packs`) and add a
`.json` file per pack:

```json
{
  "id": "my-community",
  "name": "My community",
  "category": "Custom",
  "aliases": ["name of my show"],
  "terms": ["nickname of a friend", "name of the server", "inside joke"],
  "tags": ["mycommunity"]
}
```

A custom pack with the id of a built-in pack replaces it. Built-in packs live in
`auto_video_editor/data/vocabulary/`: contributions are welcome.

---

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest
```

Project layout:

```
auto_video_editor/
  audio.py        loudness envelope (single ffmpeg pass)
  transcribe.py   Whisper backends, chunking, cache
  glossary.py     vocabulary packs, recognition, hype phrases, hashtags
  data/vocabulary 100+ packs of games and topics (JSON)
  insights.py     picture scan, content style and the other automatic decisions
  publish.py      titles, descriptions and thumbnails
  moments.py      scoring, candidate detection, selection, jump cuts
  render.py       ffmpeg filter graphs, vertical layouts, loudness, concat
  captions.py     ASS (burned-in) and SRT subtitles
  exporters.py    EDL, FCP7 XML, chapters, CSV
  vision.py       face detection for the smart crop
  pipeline.py     analyze -> project.json -> render
  cli.py / gui.py interfaces
main.py           launcher (GUI without arguments, CLI with arguments)
```

## License

MIT, see [LICENSE](LICENSE).
