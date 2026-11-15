"""Video conversion (ffmpeg) and automatic quality checks (OpenCV).

QC uses OpenCV's bundled Haar face detector, so no model downloads are needed.
It is deliberately lenient: it only catches clips that are clearly unusable
(no face visible, far too dark, far too short). You can swap in MediaPipe
later for a stricter check.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

TARGET_FPS = 30

# Thresholds. Hands-on-cheeks and side-eye can partly hide the face from a
# frontal detector, so the face threshold is intentionally low. The neutral
# lead-in and lead-out of each clip usually provide enough frontal frames.
MIN_FACE_FRACTION = 0.30
MIN_BRIGHTNESS = 45       # mean gray level, 0 to 255
MAX_BRIGHTNESS = 225
MIN_FACE_WIDTH_FRAC = 0.08  # face narrower than 8% of the frame means too far away
MIN_DURATION_FRAC = 0.6     # clip shorter than 60% of expected length

_cascade = None


def _face_cascade():
    """Find OpenCV's bundled face detector. Some OpenCV builds (e.g. conda's)
    don't expose cv2.data, so search a few known locations."""
    global _cascade
    if _cascade is None:
        name = "haarcascade_frontalface_default.xml"
        candidates = []
        if hasattr(cv2, "data"):
            candidates.append(Path(cv2.data.haarcascades) / name)
        base = Path(cv2.__file__).resolve().parent
        candidates += [base / "data" / name,
                       *base.parents[3].glob(f"share/opencv*/haarcascades/{name}")]
        for c in candidates:
            if c.exists():
                clf = cv2.CascadeClassifier(str(c))
                if not clf.empty():
                    _cascade = clf
                    break
        if _cascade is None:
            raise RuntimeError("OpenCV face detector file not found. Run: pip install opencv-python-headless")
    return _cascade


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_to_mp4(src: Path, dst: Path) -> tuple[bool, str]:
    """Browser recordings (WebM, variable frame rate) -> constant 30 fps H.264 MP4."""
    if not ffmpeg_available():
        return False, "ffmpeg not installed"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"fps={TARGET_FPS}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",  # guarantee no audio, even if a browser recorded some
        str(dst),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"
    if res.returncode != 0 or not dst.exists():
        return False, (res.stderr or "ffmpeg failed").strip()[-500:]
    return True, ""


def run_qc(video: Path, expected_seconds: float, max_samples: int = 40) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"passed": False, "flags": ["unreadable"],
                "reasons": ["The video could not be read."], "metrics": {}}

    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:  # some containers do not report a frame count
        total = int(expected_seconds * fps)
    step = max(1, total // max_samples)

    # If OpenCV's face detector is unavailable (a broken OpenCV install),
    # skip only the face check and keep the brightness and length checks.
    try:
        cascade = _face_cascade()
    except Exception:  # noqa: BLE001
        cascade = None
    brightness, face_hits, face_widths = [], 0, []
    width = height = 0
    idx = frames_read = sampled = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        frames_read += 1
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                height, width = frame.shape[:2]
                scale = 320.0 / width
                small = cv2.resize(frame, (320, int(height * scale)))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean()))
                faces = (cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                         if cascade is not None else [])
                sampled += 1
                if len(faces):
                    face_hits += 1
                    face_widths.append(max(f[2] for f in faces) / 320.0)
        idx += 1
    cap.release()

    duration = frames_read / fps if fps else 0.0
    metrics = {
        "frames": frames_read,
        "fps": round(float(fps), 2),
        "duration_s": round(duration, 2),
        "width": width,
        "height": height,
        "sampled_frames": sampled,
        "face_fraction": round(face_hits / sampled, 3) if sampled else 0.0,
        "median_face_width_frac": round(float(np.median(face_widths)), 3) if face_widths else 0.0,
        "mean_brightness": round(float(np.mean(brightness)), 1) if brightness else 0.0,
    }

    flags, reasons = [], []
    if sampled == 0:
        flags.append("no_frames")
        reasons.append("No video frames were recorded.")
    else:
        if duration < expected_seconds * MIN_DURATION_FRAC:
            flags.append("too_short")
            reasons.append("The clip is shorter than expected.")
        if metrics["mean_brightness"] < MIN_BRIGHTNESS:
            flags.append("too_dark")
            reasons.append("The video is very dark. Try turning on a light in front of you.")
        elif metrics["mean_brightness"] > MAX_BRIGHTNESS:
            flags.append("too_bright")
            reasons.append("The video is washed out. Try moving away from a bright window behind or beside you.")
        if cascade is None:
            flags.append("face_check_skipped")
        elif metrics["face_fraction"] < MIN_FACE_FRACTION:
            flags.append("no_face")
            reasons.append("We couldn't see your face in most of the clip. Make sure you're centered and facing the camera.")
        elif face_widths and metrics["median_face_width_frac"] < MIN_FACE_WIDTH_FRAC:
            flags.append("too_far")
            reasons.append("You look quite far from the camera. Try moving a little closer.")

    # "face_check_skipped" is informational only; it doesn't fail the clip.
    blocking = [f for f in flags if f != "face_check_skipped"]
    return {"passed": not blocking, "flags": flags, "reasons": reasons, "metrics": metrics}
