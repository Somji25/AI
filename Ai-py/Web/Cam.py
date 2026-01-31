import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import os
import requests

from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# =========================
# Download + Load Model
# =========================
MODEL_URL = "https://github.com/Somji25/AI/releases/download/v1.0/AI_python.h5"
MODEL_PATH = "AI_python.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub Release...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded")

# =========================
# Classes
# =========================
classes = [
    'พ่อ','ดีใจ','มีความสุข','ชอบ','ไม่สบาย','เข้าใจแล้ว','เศร้า','ยิ้ม',
    'โชคดี','หิว','ชอบ','แม่','ขอความช่วยเหลือ','ฉัน','เขา','ขอโทษ',
    'ขอโทษ','เป็นห่วง','เป็นห่วง','รัก','เขา','สวัสดี','แม่','สวัสดี',
    'เสียใจ','เสียใจ','ขอบคุณ','ยิ้ม','อิ่ม','แม่','รัก','รัก',
    'เข้าใจแล้ว','เข้าใจแล้ว','ขอความช่วยเหลือ','ห','ฬ','อ','ฮ'
]

# =========================
# Font
# =========================
font = ImageFont.truetype("Bethai.ttf", 30)

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================
# WebSocket handler
# =========================
async def process_video(websocket):
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                xs, ys = [], []
                for hand in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    h, w, _ = frame.shape
                    for lm in hand.landmark:
                        xs.append(int(lm.x * w))
                        ys.append(int(lm.y * h))

                x1, x2 = max(min(xs)-20, 0), min(max(xs)+20, w)
                y1, y2 = max(min(ys)-20, 0), min(max(ys)+20, h)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.resize(roi, (148, 148)) / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    pred = model.predict(roi, verbose=0)
                    idx = np.argmax(pred)
                    text = f"Predict: {classes[idx]} {pred[0][idx]*100:.1f}%"

                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.rectangle([x1, y1-40, x1+320, y1], fill=(0, 0, 102))
                    draw.text((x1+5, y1-35), text, font=font, fill=(255, 255, 255))
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode(".jpg", frame)
            await websocket.send(buffer.tobytes())

    cap.release()

# =========================
# Start Server
# =========================
async def main():
    async with websockets.serve(process_video, "localhost", 8765):
        print("WebSocket server running at ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
