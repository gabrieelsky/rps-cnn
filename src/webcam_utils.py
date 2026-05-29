from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional
import time

import torch
from PIL import Image


def _ordered_class_names(class_mapping: Dict[str, int]) -> List[str]:
    return [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]


def _preprocess_frame(frame_bgr, img_height: int, img_width: int) -> torch.Tensor:
    import cv2
    import numpy as np

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).resize((img_width, img_height))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict_webcam_frame(model, frame_bgr, device, img_height: int, img_width: int):
    class_input = _preprocess_frame(frame_bgr, img_height=img_height, img_width=img_width).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(class_input)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

    predicted_idx = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_idx].item())
    return predicted_idx, confidence, probabilities


def run_webcam_generalization_test(
    model,
    device,
    class_mapping: Dict[str, int],
    *,
    img_height: int,
    img_width: int,
    samples_per_class: int = 5,
    camera_index: int = 0,
    save_dir: str = "logs/webcam_generalization",
    label_keys: Optional[Dict[str, str]] = None,
    window_name: str = "RPS webcam test",
):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for the webcam test. Install it with `pip install opencv-python` "
            "or `pip install -r requirements.txt`."
        ) from exc

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    class_names = _ordered_class_names(class_mapping)
    if label_keys is None:
        label_keys = {"r": "rock", "p": "paper", "s": "scissors"}

    valid_labels = [label for label in label_keys.values() if label in class_mapping]
    if not valid_labels:
        raise ValueError("No webcam labels overlap with the model classes.")

    os.makedirs(save_dir, exist_ok=True)
    for label in valid_labels:
        os.makedirs(os.path.join(save_dir, label), exist_ok=True)

    key_to_label = {ord(key): label for key, label in label_keys.items()}
    counts = {label: 0 for label in valid_labels}
    records = []

    capture = None
    for backend in (getattr(cv2, "CAP_AVFOUNDATION", None), None):
        if backend is None:
            candidate = cv2.VideoCapture(camera_index)
        else:
            candidate = cv2.VideoCapture(camera_index, backend)

        if candidate.isOpened():
            capture = candidate
            break

        candidate.release()

    if capture is None:
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Grant camera permission to VS Code/Python and try again, "
            f"or pass a different camera_index."
        )

    print("Webcam test started.")
    print("Press r/p/s to capture the current frame as rock/paper/scissors.")
    print("Press q or Esc to finish early.")

    try:
        warmup_frames = 10
        for _ in range(warmup_frames):
            capture.read()

        while True:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    "Failed to read a frame from the webcam. Check camera permissions and make sure no other app is using it."
                )

            predicted_idx, confidence, probabilities = predict_webcam_frame(
                model=model,
                frame_bgr=frame,
                device=device,
                img_height=img_height,
                img_width=img_width,
            )
            predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else str(predicted_idx)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 110), (0, 0, 0), thickness=-1)
            cv2.putText(
                overlay,
                f"Predicted: {predicted_label} ({confidence:.2f})",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Capture: r=rock p=paper s=scissors | q=quit | target={samples_per_class} per class",
                (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            counts_text = " | ".join(f"{label}:{counts[label]}/{samples_per_class}" for label in valid_labels)
            cv2.putText(
                overlay,
                counts_text,
                (16, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key not in key_to_label:
                if all(counts[label] >= samples_per_class for label in valid_labels):
                    break
                continue

            true_label = key_to_label[key]
            if true_label not in class_mapping:
                print(f"Skipping label '{true_label}' because it is not in class_mapping.")
                continue

            sample_idx = counts[true_label]
            counts[true_label] += 1

            true_idx = int(class_mapping[true_label])
            sample_dir = os.path.join(save_dir, true_label)
            sample_path = os.path.join(
                sample_dir,
                f"{sample_idx:03d}_true_{true_label}_pred_{predicted_label}.png",
            )
            cv2.imwrite(sample_path, frame)

            records.append(
                {
                    "sample_path": sample_path,
                    "true_label": true_label,
                    "true_idx": true_idx,
                    "predicted_label": predicted_label,
                    "predicted_idx": predicted_idx,
                    "confidence": confidence,
                }
            )

            print(
                f"Captured {true_label} sample {sample_idx + 1}/{samples_per_class}: "
                f"predicted {predicted_label} ({confidence:.2f})"
            )

            if all(counts[label] >= samples_per_class for label in valid_labels):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    results = pd.DataFrame(records)
    if results.empty:
        print("No webcam samples were captured.")
        return results

    labels = [class_mapping[name] for name in class_names]
    y_true = results["true_idx"].tolist()
    y_pred = results["predicted_idx"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    summary_path = os.path.join(save_dir, "webcam_results.csv")
    results.to_csv(summary_path, index=False)

    print("\nWebcam generalization results")
    print("=" * 40)
    print(f"Captured samples: {len(results)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {summary_path}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Webcam generalization confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return results
