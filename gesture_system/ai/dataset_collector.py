"""Dataset collection utility for custom gesture classifier."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from gesture_system.config import CAMERA
from gesture_system.hand_tracker import HandTracker


def collect_dataset(label: str, samples: int = 300, output_csv: str = "gesture_dataset.csv") -> None:
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
            header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
            writer.writerow(header)

        while cap.isOpened() and captured < samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)
            hands = tracker.extract_landmarks(frame, results)

            if hands:
                lm = hands[0]["landmarks"]
                row = [p[0] for p in lm] + [p[1] for p in lm] + [label]
                writer.writerow(row)
                captured += 1
                tracker.draw(frame, hands[0])

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
