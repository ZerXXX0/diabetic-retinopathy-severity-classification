import time
import cv2
import av
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as T
import streamlit as st
import os

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# -------------------
# CONFIG
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, "model", "best_model_mobilevitv2_fine-tuning2.pth")
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative", "Ungradable"]  # change to your classes

# -------------------
# LOAD MODEL
# -------------------
@st.cache_resource
def load_model():
    model = timm.create_model("mobilevitv2_100.cvnets_in1k", num_classes=len(CLASS_NAMES))
    state_dict = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

model = load_model()

# -------------------
# PREPROCESS
# -------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------
# UI
# -------------------
st.title("📷 MobileViT v2 — Live Webcam Classifier (Square Frame)")
run_live = st.toggle("Run live classification", value=True)
interval_ms = st.slider("Infer every (ms)", min_value=100, max_value=1500, value=300, step=50)

# -------------------
# VIDEO PROCESSOR
# -------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_pred_text = ""
        self.last_time = 0.0
        self.run_live = run_live
        self.interval_ms = interval_ms
        self.one_shot = False  # trigger a single inference when requested

    def set_run_live(self, flag: bool):
        self.run_live = flag

    def set_interval_ms(self, ms: int):
        self.interval_ms = ms

    def trigger_one_shot(self):
        self.one_shot = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        size = min(h, w)
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size

        # draw square overlay
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.time()
        should_infer = self.run_live and (now - self.last_time) >= (self.interval_ms / 1000.0)
        if self.one_shot:
            should_infer = True

        if should_infer:
            crop = img[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                idx = int(torch.argmax(probs))
                conf = float(probs[idx])

            self.last_pred_text = f"{CLASS_NAMES[idx]} ({conf:.1%})"
            self.last_time = now
            self.one_shot = False

        if self.last_pred_text:
            cv2.putText(
                img,
                self.last_pred_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN server helps P2P negotiation (especially outside localhost / behind NAT)
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="mobilevitv2",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration=rtc_config,
)

# Controls wired to the active processor
if ctx.video_processor:
    ctx.video_processor.set_run_live(run_live)
    ctx.video_processor.set_interval_ms(interval_ms)

    c1, c2 = st.columns(2)
    if c1.button("📸 Capture & classify once"):
        ctx.video_processor.trigger_one_shot()
    if c2.button("⏹ Stop live"):
        ctx.video_processor.set_run_live(False)
