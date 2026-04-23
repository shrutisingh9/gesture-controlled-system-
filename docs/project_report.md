# AI-Based Gesture Controlled System for Touchless Human-Computer Interaction

## 1. Problem Statement

Traditional human-computer interaction relies on physical input devices such as mouse, keyboard, and touchscreens. These devices may be inconvenient in sterile environments, smart displays, accessibility scenarios, and hands-busy workflows. A touchless interaction system using hand gestures can provide intuitive, hygienic, and natural control of computing functions.

## 2. Objective

- Build a real-time webcam-based gesture interaction system.
- Detect hand landmarks and interpret gestures accurately.
- Map gestures to system-level controls such as cursor, click, scroll, volume, brightness, and media.
- Provide a mode-based framework (`MOUSE`, `MEDIA`, `DRAW`) for reliable interaction.
- Keep the architecture modular and extensible for AI-based custom gesture learning.

## 3. Methodology

1. Capture webcam frames using OpenCV.
2. Detect hand landmarks using MediaPipe Hands.
3. Extract key finger tip and joint coordinates.
4. Compute finger states and geometric distances (pinch, swipe, hand position).
5. Recognize gestures using rule-based thresholds and temporal cooldown windows.
6. Trigger system automation using PyAutoGUI and OS APIs.
7. Render UI overlays for gesture, mode, and FPS.

## 4. System Design Diagram

```text
Camera Input
    |
    v
Frame Preprocessing (flip, convert color)
    |
    v
MediaPipe Hand Detection
    |
    v
Landmark Extraction (21 points)
    |
    v
Gesture Recognition Engine
    |
    +--> Mode: MOUSE  --> Cursor / Click / Drag / Scroll
    +--> Mode: MEDIA  --> Volume / Brightness / Media Keys
    +--> Mode: DRAW   --> Air Canvas
    |
    v
System Action Controller (PyAutoGUI + OS hooks)
    |
    v
User Feedback Overlay (Gesture, Mode, FPS, Status)
```

## 5. Algorithm (High Level)

```text
Initialize camera, tracker, recognizer, action controller
Set mode = MOUSE, running = True

While camera is open:
    Capture frame
    Detect hand landmarks
    If no hand:
        Reset transient actions (drag, scroll anchor)
        Continue

    Select dominant hand
    Recognize gesture based on:
        - finger up/down states
        - pinch distances
        - hand motion (vertical/horizontal)
        - mode-specific rules

    If running:
        Execute mapped action with cooldowns and smoothing

    Draw landmarks and UI overlay
    Read keyboard toggles:
        s -> start/stop execution
        m -> switch mode
        c -> clear drawing canvas
        q -> exit
```

## 6. Implementation Summary

- `main.py`: real-time loop, UI overlay, mode switching, key controls.
- `gesture_system/hand_tracker.py`: MediaPipe hand detection and landmark extraction.
- `gesture_system/gesture_recognizer.py`: gesture logic and thresholds.
- `gesture_system/action_controller.py`: system automation actions.
- `gesture_system/ai/dataset_collector.py`: custom gesture data recording.
- `gesture_system/ai/train_model.py`: optional neural classifier training.

## 7. Results and Observations

- Smooth cursor control with low-pass filtering.
- Reliable click and drag through pinch-based thresholds and cooldown.
- Functional media and system controls with fallback safety for unavailable libraries.
- Real-time performance suitable for standard webcams under good lighting.

> Add screenshots from your demo run here before final submission:
> - Home screen with overlay
> - Mouse mode cursor control
> - Media mode volume/brightness control
> - Draw mode air canvas

## 8. Testing Strategy

- Tested under bright, medium, and low-light conditions.
- Measured practical FPS and interaction responsiveness.
- Evaluated false positives by introducing random hand movement.
- Tuned thresholds in `config.py` for better stability.

## 9. Edge Case Handling

- **No hand detected:** system performs no action.
- **Multiple hands:** first detected hand is selected as dominant.
- **Sudden movement:** low-pass smoothing reduces jitter.
- **Accidental gestures:** cooldown and hold-time thresholds reduce false triggers.

## 10. Future Scope

- Mobile app version for touchless phone control.
- IoT integration for smart home device commands.
- AR/VR interaction with 3D gesture space mapping.
- Voice + gesture multimodal control.
- Personalized adaptive gesture learning with user-specific calibration.

## 11. Conclusion

The project successfully demonstrates a practical and extensible touchless HCI framework using computer vision and gesture recognition. It supports real-time interaction across mouse control, media commands, system automation, and virtual drawing. The modular architecture allows future AI enhancement and domain-specific customization.
