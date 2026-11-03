import time
import io
import tempfile
import subprocess
import platform
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from pathlib import Path

try:
    from src.talking_head import generate_talking_head, sample_video_frames
except ImportError:
    from talking_head import generate_talking_head, sample_video_frames

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Emotion Lab | SOTA Facial Recognition",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM HUMAN-CENTRIC CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container tweaks */
    .main {
        background-color: #FAFAFC;
        padding: 2rem;
    }

    /* Typography */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: #0F172A !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #1E293B !important;
    }

    .sub-text {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* Cards & Containers */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* File Uploader styling */
    section[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: #6366F1;
        background: #F8FAFC;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    .sidebar-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 1rem;
    }

    /* Footer / Ethics */
    .ethics-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.85rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- MODEL INITIALIZATION ---
CLASSES = ["Happy", "Neutral", "Sad"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_emotion_model():
  model_path = Path(__file__).resolve().parents[1] / "models" / "mobilenet_emotion.pth"
  if not model_path.is_file():
    raise FileNotFoundError(
        f"Trained weights were not found: {model_path}. "
        "Run the training script first (python src/train.py --experiment all) "
        "and make sure models/mobilenet_emotion.pth is available."
    )

  model = mobilenet_v2(weights=None)
  model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
  state_dict = torch.load(model_path, map_location="cpu")
  model.load_state_dict(state_dict, strict=True)
  model.eval()
  return model

try:
  model = load_emotion_model()
except (FileNotFoundError, RuntimeError) as error:
  st.error(f"Model could not be loaded: {error}")
  st.stop()


# --- GRAD-CAM COMPUTATION ---
def compute_gradcam(input_tensor, model):
  input_tensor.requires_grad_()
  activations, gradients = [], []
  target_layer = model.features[-1]

  def forward_hook(module, input, output):
    activations.append(output)

  def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

  h_fwd = target_layer.register_forward_hook(forward_hook)
  h_bwd = target_layer.register_full_backward_hook(backward_hook)

  output = model(input_tensor)
  pred_idx = output.argmax(dim=1).item()

  model.zero_grad()
  loss = output[0, pred_idx]
  loss.backward()

  act = activations[0].detach().cpu().numpy()[0]
  grad = gradients[0].detach().cpu().numpy()[0]

  h_fwd.remove()
  h_bwd.remove()

  weights = np.mean(grad, axis=(1, 2))
  cam = np.zeros(act.shape[1:], dtype=np.float32)
  for i, w in enumerate(weights):
    cam += w * act[i]

  cam = np.maximum(cam, 0)
  if cam.max() > 0:
    cam = cam / cam.max()

  probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
  return cam, pred_idx, probs


# --- SIDEBAR ---
with st.sidebar:
  st.markdown(
      '<div class="sidebar-header">MODE SELECTION</div>', unsafe_allow_html=True
  )
  mode = st.radio(
      "Choose evaluation workflow:",
      ["Single Portrait & Grad-CAM", "Talking-Head Animation Evaluation (SadTalker)"],
      label_visibility="collapsed",
  )

  st.markdown("<br><br>", unsafe_allow_html=True)
  st.markdown(
      '<div class="sidebar-header">MODEL DETAILS</div>', unsafe_allow_html=True
  )
  st.caption("Emotion classifier: **MobileNetV2**")
  st.caption("Classes: **Happy, Neutral, Sad**")
  st.caption("Explainability: **Grad-CAM**")
  if mode == "Talking-Head Animation Evaluation (SadTalker)":
    st.caption("Animation backend: **SadTalker (VASA-1 concept alternative)**")
    st.caption("Frame sampling: **1 FPS, up to 120 frames**")

# --- MAIN CONTENT ---
st.markdown("# Facial Emotion Recognition Lab")
st.markdown(
    '<div class="sub-text">Classify facial emotions, inspect Grad-CAM attention, '
    "and evaluate frame-wise predictions on an audio-driven talking-head video.</div>",
    unsafe_allow_html=True,
)

if mode == "Single Portrait & Grad-CAM":
  st.subheader("Portrait Analysis")
  uploaded_file = st.file_uploader(
      "Drop your portrait here or click to browse",
      type=["jpg", "jpeg", "png"],
      key="single_portrait",
  )

  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    col1, col2 = st.columns(2)
    with col1:
      st.image(image, caption="Uploaded Portrait", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    start_time = time.time()
    heatmap, pred_idx, probabilities = compute_gradcam(input_tensor, model)
    inference_time = (time.time() - start_time) * 1000
    heatmap_resized = cv2.resize(
        heatmap, (image_np.shape[1], image_np.shape[0])
    )
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.65, heatmap_colored, 0.35, 0)
    with col2:
      st.image(
          overlay,
          caption="Grad-CAM Attention Map",
          use_container_width=True,
      )

    st.markdown("### Model Diagnostic Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Emotion", CLASSES[pred_idx])
    m2.metric("Confidence", f"{probabilities[pred_idx]*100:.1f}%")
    m3.metric("Latency", f"{inference_time:.1f} ms")

elif mode == "Talking-Head Animation Evaluation (SadTalker)":
  st.subheader("VASA-1-inspired Talking-Head Evaluation")
  st.warning(
      "This is a concept-level implementation using SadTalker, not Microsoft's "
      "VASA-1. SadTalker generates the talking-head video; MobileNetV2 then "
      "classifies sampled frames independently."
  )
  st.markdown(
      "Upload a portrait and speech audio. The app runs the locally installed "
      "SadTalker inference script, then produces an emotion timeline and "
      "Grad-CAM visualizations for sampled video frames."
  )

  portrait_file = st.file_uploader(
      "Source portrait", type=["jpg", "jpeg", "png"], key="talking_portrait"
  )
  audio_file = st.file_uploader(
      "Driving speech audio", type=["wav", "mp3"], key="talking_audio"
  )
  sadtalker_dir = st.text_input(
      "Path to the cloned OpenTalker/SadTalker directory",
      value="./SadTalker",
      help="The SadTalker repository and its pretrained checkpoints must already be installed locally.",
  )
  python_executable = st.text_input(
      "Python executable for the SadTalker environment",
      value="python",
      help="Use the Python executable from the dedicated SadTalker environment.",
  )
  annotations_file = st.file_uploader(
      "Optional ground-truth CSV for ΔMacro-F1",
      type=["csv"],
      key="frame_annotations",
      help="CSV columns: frame, emotion. Frame is the 1-based sampled-frame index; emotion is Happy, Neutral or Sad.",
  )
  st.caption(
      "For 6 GB GPUs, the workflow starts at 256px, batch size 1, without face enhancement. "
      "SadTalker dependencies and weights are separate from this project's requirements."
  )

  consent_given = st.checkbox(
      "I have permission to use this person's portrait and audio, and understand the generated video is synthetic.",
      value=False,
  )
  st.caption(
      "Synthetic-media notice: generated output is AI-animated and must be disclosed as synthetic. "
      "Do not use another person's face or voice without consent."
  )

  if st.button(
      "Generate video and evaluate frames",
      type="primary",
      disabled=(portrait_file is None or audio_file is None or not consent_given),
  ):
    try:
      with tempfile.TemporaryDirectory(prefix="emotion_lab_") as temp_dir:
        temp_path = Path(temp_dir)
        portrait_path = temp_path / f"portrait{Path(portrait_file.name).suffix.lower()}"
        audio_path = temp_path / f"speech{Path(audio_file.name).suffix.lower()}"
        portrait_path.write_bytes(portrait_file.getvalue())
        audio_path.write_bytes(audio_file.getvalue())

        with st.spinner("Generating talking-head video with SadTalker..."):
          generated_video, generator_log = generate_talking_head(
              sadtalker_dir=sadtalker_dir,
              python_executable=python_executable,
              portrait_path=portrait_path,
              audio_path=audio_path,
              result_dir=temp_path / "results",
              size=256,
          )

        with st.spinner("Sampling video frames and running emotion recognition + Grad-CAM..."):
          samples, video_info = sample_video_frames(
              generated_video, sample_fps=1.0, max_frames=120
          )
          frame_rows = []
          heatmap_images = []
          frame_predictions = []
          frame_inference_times = []
          for sampled_index, (timestamp, frame_rgb) in enumerate(samples, start=1):
            frame_image = Image.fromarray(frame_rgb)
            input_tensor = transform(frame_image).unsqueeze(0)
            inference_start = time.perf_counter()
            cam, pred_idx, probs = compute_gradcam(input_tensor, model)
            frame_inference_times.append((time.perf_counter() - inference_start) * 1000)
            frame_predictions.append(CLASSES[pred_idx])
            frame_rows.append({
                "Sampled frame": sampled_index,
                "Time (s)": round(timestamp, 2),
                "Happy": float(probs[0]),
                "Neutral": float(probs[1]),
                "Sad": float(probs[2]),
                "Predicted emotion": CLASSES[pred_idx],
                "Confidence": float(probs[pred_idx]),
            })

            cam_resized = cv2.resize(cam, (frame_rgb.shape[1], frame_rgb.shape[0]))
            cam_color = cv2.applyColorMap(
                np.uint8(255 * cam_resized), cv2.COLORMAP_JET
            )
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            heatmap_images.append(
                cv2.addWeighted(frame_rgb, 0.65, cam_color, 0.35, 0)
            )

          frame_df = pd.DataFrame(frame_rows)
          video_bytes = generated_video.read_bytes()
          still_image = Image.open(portrait_path).convert("RGB")
          still_tensor = transform(still_image).unsqueeze(0)
          _, still_pred_idx, _ = compute_gradcam(still_tensor, model)

          st.session_state["talking_head_video"] = video_bytes
          st.session_state["talking_head_frames"] = frame_df
          st.session_state["talking_head_heatmaps"] = heatmap_images
          video_info["avg_frame_inference_ms"] = float(np.mean(frame_inference_times))
          video_info["inference_device"] = (
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
          )
          video_info["platform"] = platform.platform()
          st.session_state["talking_head_meta"] = video_info
          st.session_state["talking_head_still_prediction"] = CLASSES[still_pred_idx]
          st.session_state["talking_head_frame_predictions"] = frame_predictions
          st.session_state["talking_head_annotations"] = (
              annotations_file.getvalue() if annotations_file is not None else None
          )
          st.success("Talking-head video generated and frame analysis completed.")

    except subprocess.TimeoutExpired:
      st.error("SadTalker exceeded the 30-minute timeout. Try a shorter audio clip.")
    except Exception as error:
      st.error(f"Talking-head evaluation failed: {error}")

  if "talking_head_video" in st.session_state:
    st.markdown("### Generated talking-head video")
    st.video(st.session_state["talking_head_video"])
    metadata = st.session_state["talking_head_meta"]
    st.info("AI-GENERATED SYNTHETIC MEDIA — SadTalker animation, not an authentic recording.")
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Video duration", f"{metadata['duration_seconds']:.1f} s")
    info2.metric("Source FPS", f"{metadata['fps']:.1f}")
    info3.metric("Analyzed frames", str(metadata["sampled_frames"]))
    info4.metric("Avg. frame inference", f"{metadata['avg_frame_inference_ms']:.1f} ms")
    st.caption(
        f"Classifier device: {metadata['inference_device']} · "
        f"Environment: {metadata['platform']}"
    )

    frame_df = st.session_state["talking_head_frames"]
    st.markdown("### Per-frame emotion timeline")
    chart_df = frame_df.set_index("Time (s)")[["Happy", "Neutral", "Sad"]]
    st.line_chart(chart_df)
    st.dataframe(
        frame_df[["Sampled frame", "Time (s)", "Predicted emotion", "Confidence"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Grad-CAM attention maps")
    heatmaps = st.session_state["talking_head_heatmaps"]
    selected_indices = np.linspace(
        0, len(heatmaps) - 1, num=min(6, len(heatmaps)), dtype=int
    )
    heatmap_cols = st.columns(min(3, len(selected_indices)))
    for position, frame_index in enumerate(selected_indices):
      with heatmap_cols[position % len(heatmap_cols)]:
        st.image(
            heatmaps[frame_index],
            caption=f"Sampled frame {frame_index + 1}",
            use_container_width=True,
        )

    st.markdown("### Still-image vs animated-video evaluation")
    st.metric(
        "Still-image prediction",
        st.session_state["talking_head_still_prediction"],
    )
    annotation_bytes = st.session_state.get("talking_head_annotations")
    if annotation_bytes:
      from sklearn.metrics import f1_score
      try:
        annotation_df = pd.read_csv(io.BytesIO(annotation_bytes))
        required_columns = {"frame", "emotion"}
        if not required_columns.issubset(annotation_df.columns):
          raise ValueError("CSV must contain columns named 'frame' and 'emotion'.")
        annotation_df["emotion"] = annotation_df["emotion"].astype(str).str.strip().str.title()
        if not set(annotation_df["emotion"]).issubset(set(CLASSES)):
          raise ValueError("Emotion labels must be Happy, Neutral or Sad.")
        annotation_df["frame"] = pd.to_numeric(annotation_df["frame"], errors="raise").astype(int)
        if annotation_df["frame"].duplicated().any():
          raise ValueError("Each sampled frame index may appear only once in the CSV.")
        if not set(annotation_df["frame"]).issubset(set(frame_df["Sampled frame"])):
          raise ValueError("CSV frame indices must refer to sampled frames shown in this result.")

        indexed_predictions = frame_df.set_index("Sampled frame")["Predicted emotion"]
        evaluated = annotation_df.sort_values("frame")
        y_true = evaluated["emotion"].tolist()
        y_video = indexed_predictions.loc[evaluated["frame"]].tolist()
        y_still = [st.session_state["talking_head_still_prediction"]] * len(y_true)
        labels = CLASSES
        still_f1 = f1_score(y_true, y_still, labels=labels, average="macro", zero_division=0)
        video_f1 = f1_score(y_true, y_video, labels=labels, average="macro", zero_division=0)
        delta_f1 = video_f1 - still_f1
        score1, score2, score3 = st.columns(3)
        score1.metric("Still-image Macro-F1", f"{still_f1:.3f}")
        score2.metric("Video-frame Macro-F1", f"{video_f1:.3f}")
        score3.metric("ΔMacro-F1 (video − still)", f"{delta_f1:+.3f}")
        st.caption(
            f"Scores use {len(evaluated)} manually annotated sampled frames. "
            "The same frame labels are used for both conditions; this is not a "
            "claim about VASA-1 and does not replace held-out test-set evaluation."
        )
      except Exception as error:
        st.error(f"Could not calculate ΔMacro-F1: {error}")
    else:
      st.info(
          "Upload a CSV with columns 'frame,emotion' and rerun the analysis to "
          "calculate still-image Macro-F1, video-frame Macro-F1 and ΔMacro-F1. "
          "Without independent frame labels, only prediction consistency can be reported."
      )
      predictions = st.session_state["talking_head_frame_predictions"]
      consistency = (
          np.asarray(predictions) == predictions[0]
      ).mean() * 100
      st.metric("Video prediction consistency", f"{consistency:.1f}%")
      st.caption(
          "Consistency measures agreement with the first sampled frame prediction; "
          "it is not Accuracy or F1-score."
      )

# --- FOOTER ---
st.markdown(
    '<div class="ethics-footer"><b>Privacy notice:</b> Uploaded images are processed'
    " by the machine running this Streamlit app. If the app is deployed remotely,"
    " images are sent to that server; do not upload images without appropriate"
    " consent. This app does not intentionally save uploaded images or create"
    " biometric identity profiles.</div>",
    unsafe_allow_html=True,
)

#python3 -m streamlit run app/app.py --server.address 127.0.0.1