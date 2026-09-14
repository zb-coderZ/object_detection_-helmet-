# Smart Helmet Detection — YOLO vs SSD

An object detection project that trains and compares two architectures, YOLOv8 and SSD (SSDLite MobileNetV3), on a smart helmet compliance dataset, with a public web app for live inference.

**Live demo:** https://zohaib-object-detection.streamlit.app/

Built by **HAYTHIX AI** — DevOps & AI Automation

---

## Overview

The goal was to detect motorcycle riders, passengers, bikes, and helmet compliance status from road images, using two different object detection approaches trained on the same dataset, then compare their performance and ship both behind a single interactive UI.

## Dataset

**Source:** Smart Helmet Detection using DL (Kaggle)

**Classes (7):**
| ID | Class |
|----|-------|
| 0 | driver_with_helmet |
| 1 | bike |
| 2 | driver |
| 3 | passenger_with_helemt |
| 4 | passenger |
| 5 | driver_without_helmet |
| 6 | passenger_without_helemt |

**Split:**
| Set | Images | Labels |
|-----|--------|--------|
| Train | 366 | 366 |
| Validation | 65 | 65 |
| Test | 52 | 52 |

Two originally unlabeled training images were excluded during cleanup.

## Models

### YOLOv8n
- Framework: Ultralytics 8.4.150
- Hardware: Tesla T4 GPU
- Training: 25 epochs, image size 640, batch size 16

### SSD (SSDLite320 MobileNetV3-Large)
- Framework: PyTorch 2.10.0 + torchvision 0.25.0 (CUDA)
- Custom classification head sized for 8 classes (7 + background)
- Optimizer: SGD (lr 0.001, momentum 0.9, weight decay 0.0005)
- Scheduler: StepLR (step size 8, gamma 0.1)
- Training: 20 epochs
- Annotations converted from YOLO TXT format to Pascal VOC XML for training

A key architecture detail: the trained SSD checkpoint used a reduced-tail MobileNetV3 backbone. Reloading it requires `weights_backbone=None` alongside `weights=None` when reconstructing the model, otherwise the classification head channel dimensions mismatch against the checkpoint.

## Results

| Metric | YOLOv8n | SSD |
|--------|---------|-----|
| Precision | 74.50% | N/A |
| Recall | 71.00% | N/A |
| mAP50 | 75.80% | 42.84% |
| mAP50-95 | 39.80% | 19.56% |

**Per-class mAP50 (YOLOv8n):**
| Class | mAP50 |
|-------|-------|
| driver_with_helmet | 0.852 |
| bike | 0.873 |
| driver | 0.853 |
| passenger_with_helemt | 0.426 |
| passenger | 0.879 |
| driver_without_helmet | 0.756 |
| passenger_without_helemt | 0.670 |

**Conclusion:** YOLOv8n outperformed SSD across all shared metrics on this dataset. SSD precision/recall were not computed to avoid reporting invented figures; only the metrics that were actually calculated are shown.

## Application

A single web app lets a user upload an image, choose YOLO or SSD, and view bounding boxes, detected classes, and confidence scores side by side with the uploaded image.

**Tech stack:**
- Streamlit (UI)
- PyTorch / Ultralytics (inference)
- OpenCV (bounding box rendering for SSD output)

**Branding:** Charcoal navy (#0F1620) and amber gold (#E8A33D), matching the HAYTHIX AI visual identity.

## Deployment

The app is deployed on **Streamlit Community Cloud**, connected directly to this GitHub repository. Any push to the `main` branch triggers an automatic redeploy.

Hugging Face Spaces and Render.com were evaluated first but required a paid plan on the accounts used for this project (Gradio/Docker Spaces on Hugging Face, and card verification on Render's free web service tier at the time of deployment). Streamlit Community Cloud was chosen as the free, no-card alternative.

## Project Structure

```
.
├── app.py              # Streamlit application (UI + inference)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition (used during the Render evaluation)
├── best_yolo.pt         # Trained YOLOv8n weights
├── best_ssd.pth         # Trained SSD weights
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running Globally
```browser
https://zohaib-object-detection.streamlit.app/
```
