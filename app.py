
import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

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


def predict(image, model_choice):
    if image is None:
        return None, "Please upload an image."

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

    detection_text = "\n".join(detections) if detections else "No objects detected."
    return output_image, detection_text


CUSTOM_CSS = """
:root {
    --brand-navy: #0F1620;
    --brand-gold: #E8A33D;
}

.gradio-container {
    background: var(--brand-navy) !important;
    font-family: "Inter", "Segoe UI", sans-serif !important;
}

#header-block {
    text-align: center;
    padding: 28px 12px 8px 12px;
}

#header-block h1 {
    color: var(--brand-gold) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    margin-bottom: 4px !important;
}

#header-block p {
    color: #B8C2CC !important;
    font-size: 0.95rem !important;
}

#brand-tag {
    color: #6B7684 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.gr-panel, .block {
    background: #161F2C !important;
    border: 1px solid #232E3D !important;
    border-radius: 14px !important;
}

button.primary {
    background: var(--brand-gold) !important;
    color: var(--brand-navy) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
}

button.primary:hover {
    opacity: 0.88 !important;
}

label span {
    color: #D6DCE2 !important;
    font-weight: 500 !important;
}

textarea, input {
    background: #0F1620 !important;
    color: #E8A33D !important;
    border: 1px solid #232E3D !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(title="HAYTHIX AI — Smart Helmet Detection", css=CUSTOM_CSS) as demo:

    with gr.Column(elem_id="header-block"):
        gr.Markdown("<p id='brand-tag'>HAYTHIX AI</p>")
        gr.Markdown("# Smart Helmet Detection")
        gr.Markdown("Upload a road image and choose YOLO or SSD to detect riders, passengers, bikes, and helmet compliance in real time.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            model_choice = gr.Radio(
                ["YOLO", "SSD"],
                value="YOLO",
                label="Detection Model"
            )
            predict_button = gr.Button("Run Detection", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Detection Result")
            output_text = gr.Textbox(label="Detected Objects", lines=10)

    gr.Markdown(
        "<p style='text-align:center; color:#6B7684; font-size:0.8rem; margin-top:20px;'>"
        "Built by HAYTHIX AI — DevOps & AI Automation</p>"
    )

    predict_button.click(
        fn=predict,
        inputs=[image_input, model_choice],
        outputs=[output_image, output_text]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
