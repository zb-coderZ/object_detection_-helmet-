
import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

st.set_page_config(page_title="HAYTHIX AI — Smart Helmet Detection", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {
    background-color: #0F1620;
}
h1, h2, h3 {
    color: #E8A33D !important;
}
p, label, .stMarkdown {
    color: #D6DCE2 !important;
}
.stButton>button {
    background-color: #E8A33D;
    color: #0F1620;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.4em;
}
.stButton>button:hover {
    opacity: 0.88;
    color: #0F1620;
}
[data-testid="stSidebar"] {
    background-color: #161F2C;
}
.block-container {
    padding-top: 2rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "driver_with_helmet",
    "bike",
    "driver",
    "passenger_with_helemt",
    "passenger",
    "driver_without_helmet",
    "passenger_without_helemt"
]


@st.cache_resource
def load_models():
    yolo_model = YOLO("best_yolo.pt")

    ssd_model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=8
    )
    ssd_model.load_state_dict(
        torch.load("best_ssd.pth", map_location=device, weights_only=True)
    )
    ssd_model.to(device)
    ssd_model.eval()

    return yolo_model, ssd_model


yolo_model, ssd_model = load_models()


def predict(image, model_choice):
    if model_choice == "YOLO":
        results = yolo_model.predict(source=image, conf=0.30, verbose=False)
        result = results[0]
        output_image = result.plot()

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append(f"{CLASS_NAMES[cls_id]}  —  {confidence:.2%}")

    else:
        image_tensor = TF.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = ssd_model([image_tensor])[0]

        output_image = np.array(image).copy()
        detections = []

        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if score < 0.30:
                continue

            box = box.int().cpu().numpy()
            x1, y1, x2, y2 = box
            label_id = int(label) - 1
            confidence = float(score)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (232, 163, 61), 3)
            text = f"{CLASS_NAMES[label_id]} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(output_image, (x1, y1 - th - 12), (x1 + tw + 6, y1), (232, 163, 61), -1)
            cv2.putText(
                output_image, text, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 22, 32), 2
            )

            detections.append(f"{CLASS_NAMES[label_id]}  —  {confidence:.2%}")

    return output_image, detections


st.markdown("<p style='letter-spacing:2px; color:#6B7684; text-transform:uppercase;'>HAYTHIX AI</p>", unsafe_allow_html=True)
st.title("Smart Helmet Detection")
st.write("Upload a road image and choose YOLO or SSD to detect riders, passengers, bikes, and helmet compliance.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    model_choice = st.radio("Detection Model", ["YOLO", "SSD"], horizontal=True)
    run_button = st.button("Run Detection")

with col2:
    if uploaded_file and run_button:
        image = Image.open(uploaded_file).convert("RGB")
        output_image, detections = predict(image, model_choice)

        st.image(output_image, caption="Detection Result", use_container_width=True)

        if detections:
            st.text_area("Detected Objects", "\n".join(detections), height=200)
        else:
            st.info("No objects detected.")
    elif not uploaded_file:
        st.write("Upload an image and click Run Detection to see results here.")

st.markdown(
    "<p style='text-align:center; color:#6B7684; font-size:0.8rem; margin-top:2rem;'>Built by HAYTHIX AI — DevOps & AI Automation</p>",
    unsafe_allow_html=True
)
