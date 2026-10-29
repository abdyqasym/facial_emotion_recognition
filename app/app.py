import time
import cv2
import numpy as np
import PIL.Image as Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Facial Emotion Recognition", page_icon="🎭", layout="centered"
)

st.title("🎭 Facial Emotion Recognition + Grad-CAM")
st.write(
    "Upload a facial image to classify the emotion and visualize model"
    " attention using Grad-CAM heatmaps."
)

# --- CONFIG & CONSTANTS ---
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
    st.sidebar.success("Loaded fine-tuned model weights successfully!")
  except FileNotFoundError:
    st.sidebar.warning(
        "Custom weights not found in 'models/'. Using default model architecture."
    )

  model.eval()
  return model


model = load_emotion_model()


# --- GRAD-CAM COMPUTATION ---
def compute_gradcam(input_tensor, model):
  # Set requires_grad to True for input tensor
  input_tensor.requires_grad_()

  activations = []
  gradients = []

  target_layer = model.features[-1]

  def forward_hook(module, input, output):
    activations.append(output)

  def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

  h_fwd = target_layer.register_forward_hook(forward_hook)
  h_bwd = target_layer.register_full_backward_hook(backward_hook)

  # Forward pass
  output = model(input_tensor)
  pred_idx = output.argmax(dim=1).item()

  # Backward pass
  model.zero_grad()
  loss = output[0, pred_idx]
  loss.backward()

  # Retrieve saved activations and gradients
  act = activations[0].detach().cpu().numpy()[0]
  grad = gradients[0].detach().cpu().numpy()[0]

  # Cleanup hooks
  h_fwd.remove()
  h_bwd.remove()

  # Calculate weights
  weights = np.mean(grad, axis=(1, 2))
  cam = np.zeros(act.shape[1:], dtype=np.float32)

  for i, w in enumerate(weights):
    cam += w * act[i]

  cam = np.maximum(cam, 0)
  if cam.max() > 0:
    cam = cam / cam.max()

  probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
  return cam, pred_idx, probs


# --- USER INTERFACE ---
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

  # Run Grad-CAM & Prediction
  heatmap, pred_idx, probabilities = compute_gradcam(input_tensor, model)

  inference_time = (time.time() - start_time) * 1000

  # Overlay heatmap
  heatmap_resized = cv2.resize(
      heatmap, (image_np.shape[1], image_np.shape[0])
  )
  heatmap_colored = cv2.applyColorMap(
      np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
  )
  heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
  overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

  pred_label = CLASSES[pred_idx]
  confidence = probabilities[pred_idx] * 100

  with col2:
    st.image(overlay, caption="Grad-CAM Explanation", use_column_width=True)

  st.write("---")
  st.subheader("Prediction Results")

  res_col1, res_col2 = st.columns(2)
  with res_col1:
    st.metric(label="Predicted Emotion", value=pred_label)
  with res_col2:
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

  st.caption(f"⚡ **Inference Latency:** {inference_time:.2f} ms")

  st.subheader("Class Probabilities")
  for cls_name, prob in zip(CLASSES, probabilities):
    st.progress(float(prob), text=f"{cls_name}: {prob*100:.1f}%")

  st.write("---")
  st.caption(
      "⚠️ **Ethics Disclosure:** Facial data processed in real-time."
      " Synthetic images used in testing are explicitly tagged."
  )