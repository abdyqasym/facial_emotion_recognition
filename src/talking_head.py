"""SadTalker adapter for the VASA-1-inspired talking-head evaluation workflow.

SadTalker is an external, separately installed research implementation. This
module invokes its documented inference.py CLI; it does not vendor model code
or download model weights into this repository.
"""
from pathlib import Path
import subprocess

import cv2


def generate_talking_head(
    sadtalker_dir,
    python_executable,
    portrait_path,
    audio_path,
    result_dir,
    size=256,
    timeout_seconds=1800,
):
    """Run SadTalker CLI and return the generated MP4 path."""
    repo = Path(sadtalker_dir).expanduser().resolve()
    inference_script = repo / "inference.py"
    checkpoint_dir = repo / "checkpoints"
    if not inference_script.is_file():
        raise FileNotFoundError(
            f"SadTalker inference.py was not found in {repo}. "
            "Clone OpenTalker/SadTalker and install its dependencies first."
        )
    if not checkpoint_dir.is_dir() or not any(checkpoint_dir.iterdir()):
        raise FileNotFoundError(
            f"SadTalker checkpoints are missing in {checkpoint_dir}. "
            "Follow the model-download instructions in the SadTalker README."
        )

    output = Path(result_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        str(inference_script),
        "--driven_audio", str(Path(audio_path).resolve()),
        "--source_image", str(Path(portrait_path).resolve()),
        "--result_dir", str(output),
        "--size", str(size),
        "--batch_size", "1",
        "--preprocess", "crop",
        "--still",
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "No process output")[-8000:]
        raise RuntimeError(
            "SadTalker inference failed. Check its separate Python environment, "
            "CUDA/PyTorch installation, FFmpeg and checkpoints.\n" + details
        )

    videos = sorted(
        output.rglob("*.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        details = (completed.stdout or "")[-4000:]
        raise RuntimeError(
            "SadTalker finished without producing an MP4 file.\n" + details
        )
    return videos[0], completed.stdout


def sample_video_frames(video_path, sample_fps=1.0, max_frames=120):
    """Return evenly sampled (timestamp_seconds, RGB frame) pairs."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open generated video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if frame_count else 0.0
    stride = max(1, int(round(source_fps / max(sample_fps, 0.1))))
    samples = []
    frame_index = 0

    while len(samples) < max_frames:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            timestamp = frame_index / source_fps
            samples.append((timestamp, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        frame_index += 1

    capture.release()
    if not samples:
        raise RuntimeError("No frames could be decoded from the generated video.")
    metadata = {
        "fps": float(source_fps),
        "frame_count": frame_count,
        "duration_seconds": float(duration),
        "sampled_frames": len(samples),
        "sampling_fps": float(source_fps / stride),
    }
    return samples, metadata
