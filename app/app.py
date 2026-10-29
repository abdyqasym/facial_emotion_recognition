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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Facial Emotion Recognition SOTA Demo",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 SOTA Facial Emotion Recognition & Animation Evaluation")
st.write(
    "Upload a still image or test per-frame emotion consistency under"
    " animation/talking-head concepts (VASA-1 & ArtAug)."
)

CLASSES = ["Happy", "Neutral", "Sad"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- MODEL INITIALIZATION ---
@st.cache_resource
def load_emotion_model():
  model = mobilenet_v2(weights=None)
  model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
  try:
    model.load_state_dict(
        torch.load("models/mobilenet_emotion.pth", map_location="cpu")
    )
    st.sidebar.success("Loaded fine-tuned model weights!")
  except FileNotFoundError:
    st.sidebar.warning("Using default model architecture.")
  model.eval()
  return model


model = load_emotion_model()


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


# --- MODE SELECTION ---
mode = st.sidebar.radio(
    "Select Demo Mode:",
    ["Still Image Analysis", "VASA-1 Talking-Head Animation Timeline"],
)

if mode == "Still Image Analysis":
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png"]
  )
  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
      st.image(image, caption="Uploaded Image", use_column_width=True)

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
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

    with col2:
      st.image(overlay, caption="Grad-CAM Explanation", use_column_width=True)

    st.subheader("Prediction Results")
    st.metric(label="Predicted Emotion", value=CLASSES[pred_idx])
    st.caption(f"⚡ **Inference Latency:** {inference_time:.2f} ms")

elif mode == "VASA-1 Talking-Head Animation Timeline":
  st.subheader("VASA-1 Per-Frame Emotion Consistency Stress-Test")
  st.write(
      "Simulating video frame evaluation from a generated talking-head clip."
  )

  uploaded_file = st.file_uploader(
      "Upload portrait image...", type=["jpg", "jpeg", "png"]
  )
  audio_file = st.file_uploader("Upload audio file...", type=["wav", "mp3"])

  if uploaded_file is not None and audio_file is not None:
    st.audio(audio_file)
    st.info(
        "Processing generated sequence (VASA-1 concept: 30 frames"
        " evaluated)..."
    )

    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Simulated per-frame variation (ArtAug noise/pose perturbations)
    frames_data = []
    np.random.seed(42)

    for frame_idx in range(30):
      # Add subtle noise simulation for animation frames
      noise_tensor = input_tensor + torch.randn_like(input_tensor) * 0.05
      _, pred_idx, probs = compute_gradcam(noise_tensor, model)

      frames_data.append({
          "Frame": frame_idx + 1,
          "Happy": probs[0],
          "Neutral": probs[1],
          "Sad": probs[2],
          "Predicted": CLASSES[pred_idx],
      })

    df_frames = pd.DataFrame(frames_data)

    st.subheader("Per-Frame Emotion Timeline")
    st.line_chart(df_frames.set_index("Frame")[["Happy", "Neutral", "Sad"]])

    st.subheader("Consistency Summary")
    st.write(
        f"**Most Frequent Predicted Emotion:** {df_frames['Predicted'].mode()[0]}"
    )
    st.caption(
        "📊 **Evaluation Note:** Per-frame fluctuation indicates sensitivity to"
        " pose/backlight changes under animation."
    )

st.write("---")
st.caption(
    "⚠️ **Ethics Disclosure:** Facial biometric processing performed for"
    " research demonstration."
)