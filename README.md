# AI-Based Gesture Controlled System for Touchless Human-Computer Interaction

A professional final-year project that enables touchless control of a computer using real-time hand gestures captured through a webcam.

## Features

- Real-time hand landmark detection using MediaPipe
- Smooth cursor movement using index finger tracking
- Gesture-based left click, right click, double click, drag-and-drop
- Scroll control with two-finger movement
- Volume and brightness control
- Media controls (play/pause, next, previous)
- App launch, app close, screenshot, and system lock gestures
- Virtual drawing mode (air canvas)
- Mode switching: `MOUSE`, `MEDIA`, and `DRAW`
- Optional AI workflow for dataset collection and model training

## Project Architecture

`Camera Input -> Frame Processing -> Hand Detection -> Landmark Extraction -> Gesture Recognition -> Action Mapping -> System Control`

## Folder Structure

```text
gesture controlled system/
├── main.py
├── requirements.txt
├── README.md
├── gesture_system/
│   ├── __init__.py
│   ├── config.py
│   ├── hand_tracker.py
│   ├── gesture_recognizer.py
│   ├── action_controller.py
│   ├── utils.py
│   └── ai/
│       ├── __init__.py
│       ├── dataset_collector.py
│       └── train_model.py
└── docs/
    └── project_report.md
```

## Setup

> Recommended Python version: `3.10` or `3.11` (MediaPipe may fail on `3.13+`).

1. Create and activate virtual environment:
   - Windows (PowerShell):
     - `py -3.11 -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
   - Optional AI training stack: `pip install -r requirements-ai.txt`
3. Run the project:
   - `python main.py`

## Controls

- `s`: start/stop gesture execution
- `m`: switch mode (`MOUSE -> MEDIA -> DRAW -> MOUSE`)
- `q`: quit application
- `c` in draw mode: clear virtual canvas

## Gesture Mapping (Default)

### Mouse Mode
- Index up: move cursor
- Thumb + index pinch: left click
- Thumb + middle pinch: right click
- Fast repeated thumb + index pinch: double click
- Pinch hold + move: drag
- Index + middle up: scroll by vertical movement

### Media Mode
- Thumb-index distance: volume
- Hand vertical position: brightness
- Three-finger gesture: play/pause
- Swipe right gesture: next track
- Swipe left gesture: previous track

### Draw Mode
- Index finger up: draw
- Fist (all fingers down): stop drawing
- `c`: clear canvas

### System / Utility Gestures
- Open app gesture (custom hook)
- Close active app gesture
- Screenshot gesture
- Lock system gesture

## Notes

- Some system actions are OS-specific.
- If optional packages for media/brightness are unavailable, the app falls back gracefully and keeps running.
- Adjust thresholds in `gesture_system/config.py` for your camera and lighting conditions.
