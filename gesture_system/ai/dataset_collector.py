"""Dataset collection utility for custom gesture classifier."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from gesture_system.config import CAMERA
from gesture_system.hand_tracker import HandTracker


def _normalize_landmarks(lm: list[tuple[int, int]]) -> list[float]:
    # Normalize landmarks relative to wrist so model is less sensitive to position/scale.
    base_x, base_y = lm[0]
    rel = [(x - base_x, y - base_y) for x, y in lm]
    max_abs = max(max(abs(x), abs(y)) for x, y in rel) or 1
    return [x / max_abs for x, _ in rel] + [y / max_abs for _, y in rel]


def collect_dataset(label: str, samples: int = 300, output_csv: str = "gesture_dataset.csv") -> None:
    # Collect labeled gesture samples from webcam and append to CSV.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
    tracker = HandTracker(CAMERA.detection_confidence, CAMERA.tracking_confidence)

    output_path = Path(output_csv)
    write_header = not output_path.exists()
    captured = 0

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label", "lighting"]
            writer.writerow(header)

        while cap.isOpened() and captured < samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            # Optional enhanced frame variant for low-light robustness.
            enhanced = frame
            if CAMERA.low_light_enhancement:
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
                enhanced = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

            results = tracker.process(frame)
            hands = tracker.extract_landmarks(frame, results)
            results_low = tracker.process(enhanced)
            hands_low = tracker.extract_landmarks(enhanced, results_low)

            if hands:
                # Save normal-light sample.
                lm = hands[0]["landmarks"]
                row = _normalize_landmarks(lm) + [label, "normal"]
                writer.writerow(row)
                captured += 1
                tracker.draw(frame, hands[0])
            if hands_low and captured < samples:
                # Save enhanced-light sample variant.
                lm_low = hands_low[0]["landmarks"]
                row_low = _normalize_landmarks(lm_low) + [label, "enhanced"]
                writer.writerow(row_low)
                captured += 1

            cv2.putText(frame, f"Label: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
            cv2.putText(
                frame,
                f"Captured: {captured}/{samples}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 200, 0),
                2,
            )
            cv2.imshow("Dataset Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_dataset(label="custom_gesture")
