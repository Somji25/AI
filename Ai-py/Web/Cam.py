# ================= SYSTEM =================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import asyncio
import websockets
import numpy as np
import requests
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

# ================= MODEL =================
MODEL_URL = "https://github.com/Somji25/AI/releases/download/v1.0/model_tf_new.keras"
MODEL_PATH = "AI_python.h5"

if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded")

classes = [
    'พ่อ','ดีใจ','มีความสุข','ชอบ','ไม่สบาย','เข้าใจแล้ว','เศร้า','ยิ้ม',
    'โชคดี','หิว','แม่','ขอความช่วยเหลือ','ฉัน','เขา','ขอโทษ','เป็นห่วง',
    'รัก','สวัสดี','เสียใจ','ขอบคุณ','อิ่ม','ห','ฬ','อ','ฮ'
]

font = ImageFont.truetype("Bethai.ttf", 28)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= WEBSOCKET =================
async def process_video(ws):
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        async for message in ws:
            img = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            img = cv2.resize(img, (480, 360))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                xs, ys = [], []
                h, w, _ = img.shape

                for hand in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                    for lm in hand.landmark:
                        xs.append(int(lm.x * w))
                        ys.append(int(lm.y * h))

                x1, x2 = max(min(xs)-20, 0), min(max(xs)+20, w)
                y1, y2 = max(min(ys)-20, 0), min(max(ys)+20, h)

                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.resize(roi, (148, 148)) / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    pred = model.predict(roi, verbose=0)
                    idx = np.argmax(pred)

                    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil)
                    draw.rectangle([x1, y1-35, x1+300, y1], fill=(0, 0, 120))
                    draw.text(
                        (x1+5, y1-30),
                        f"{classes[idx]} {pred[0][idx]*100:.1f}%",
                        font=font,
                        fill=(255,255,255)
                    )
                    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode(".jpg", img)
            await ws.send(buffer.tobytes())

# ================= MAIN =================
async def main():
    PORT = int(os.environ.get("PORT", 10000))
    async with websockets.serve(process_video, "0.0.0.0", PORT):
        print("WebSocket running on port", PORT)
        await asyncio.Future()

asyncio.run(main())


