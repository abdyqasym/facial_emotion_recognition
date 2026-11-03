import time
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
      ["Single Portrait & Grad-CAM", "Noise Robustness Test (Timeline)"],
      label_visibility="collapsed",
  )

  st.markdown("<br><br>", unsafe_allow_html=True)
  st.markdown(
      '<div class="sidebar-header">MODEL DETAILS</div>', unsafe_allow_html=True
  )
  st.caption("Backbone: **MobileNetV2**")
  st.caption("Classes: **Happy, Neutral, Sad**")
  st.caption("Explainability: **Grad-CAM Layer 18**")

# --- MAIN CONTENT ---
st.markdown("# Facial Emotion Recognition Lab")
st.markdown(
    '<div class="sub-text">Explore real-time emotion classification,'
    " visual explainability via Grad-CAM, and per-frame dynamic consistency"
    " stress tests.</div>",
    unsafe_allow_html=True,
)

if mode == "Single Portrait & Grad-CAM":
  st.subheader("Portrait Analysis")

  uploaded_file = st.file_uploader(
      "Drop your portrait here or click to browse", type=["jpg", "jpeg", "png"]
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
    m2.metric(
        "Confidence", f"{probabilities[pred_idx]*100:.1f}%"
    )
    m3.metric("Latency", f"{inference_time:.1f} ms")

elif mode == "Noise Robustness Test (Timeline)":
  st.subheader("Noise Robustness Test")
  st.caption(
      "This experiment adds small Gaussian noise to one uploaded image and "
      "measures whether the model keeps its original prediction. It is not a "
      "VASA-1 animation or a ground-truth accuracy evaluation."
  )

  uploaded_file = st.file_uploader(
      "Upload Portrait Image", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    _, reference_idx, _ = compute_gradcam(input_tensor.clone(), model)
    predictions = []
    frames_data = []

    # Fixed seed makes the perturbation sequence reproducible.
    generator = torch.Generator(device="cpu").manual_seed(42)
    for frame_idx in range(30):
      noise = torch.randn(
          input_tensor.shape, generator=generator, dtype=input_tensor.dtype
      ) * 0.04
      noise_tensor = input_tensor + noise
      _, pred_idx, probs = compute_gradcam(noise_tensor, model)
      predictions.append(pred_idx)

      frames_data.append({
          "Frame": frame_idx + 1,
          "Happy": float(probs[0]),
          "Neutral": float(probs[1]),
          "Sad": float(probs[2]),
          "Predicted": CLASSES[pred_idx],
      })

    df_frames = pd.DataFrame(frames_data)
    stability_rate = (
        np.asarray(predictions) == reference_idx
    ).mean() * 100

    st.markdown("### Prediction Probability Under Input Perturbations")
    st.line_chart(df_frames.set_index("Frame")[["Happy", "Neutral", "Sad"]])

    st.markdown("### Robustness Metrics")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Original Prediction", CLASSES[reference_idx])
    res_col2.metric("Prediction Consistency", f"{stability_rate:.1f}%")
    st.caption(
        "Consistency is the share of noisy inputs that retain the prediction "
        "made for the clean image. It is not Accuracy or F1-score because no "
        "independent ground-truth labels are supplied."
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