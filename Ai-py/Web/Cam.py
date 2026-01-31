import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from PIL import ImageFont, ImageDraw, Image
import gdown
import io
import h5py
import os

# ====== Load model from Google Drive (ไม่ต้องเซฟไฟล์) ======
file_id = "15qEX09mZlmeXnMmu3SUxkU0X7rU1Bmsy"
url = f"https://drive.google.com/uc?id={file_id}"

# ดาวน์โหลดไฟล์มาเป็น bytes
response_path = "AI_python.h5"
if not os.path.exists(response_path):
    gdown.download(url, response_path, quiet=False)

# โหลดโมเดลจากไฟล์ที่ดาวน์โหลดมา
model = load_model(response_path)

# ====== Classes และ Font ======
classes = ['พ่อ','ดีใจ','มีความสุข','ชอบ','ไม่สบาย','เข้าใจแล้ว','เศร้า','ยิ้ม',
           'โชคดี','หิว','ชอบ','แม่','ขอความช่วยเหลือ','ฉัน','เขา','ขอโทษ',
           'ขอโทษ','เป็นห่วง','เป็นห่วง','รัก','เขา','สวัสดี','แม่','สวัสดี',
           'เสียใจ','เสียใจ','ขอบคุณ','ยิ้ม','อิ่ม','แม่','รัก','รัก',
           'เข้าใจแล้ว','เข้าใจแล้ว','ขอความช่วยเหลือ','ห','ฬ','อ','ฮ']

font = ImageFont.truetype("Bethai.ttf", 30)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ====== WebSocket handler ======
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

                x1, x2 = max(min(xs)-20,0), min(max(xs)+20,w)
                y1, y2 = max(min(ys)-20,0), min(max(ys)+20,h)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.resize(roi, (148,148)) / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    pred = model.predict(roi, verbose=0)
                    idx = np.argmax(pred)
                    text = f"ทำนาย: {classes[idx]} {pred[0][idx]*100:.1f}%"

                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.rectangle([x1, y1-40, x1+300, y1], fill=(0,0,102))
                    draw.text((x1+5, y1-35), text, font=font, fill=(255,255,255))
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode(".jpg", frame)
            await websocket.send(buffer.tobytes())

    cap.release()

# ====== Start server ======
async def main():
    async with websockets.serve(process_video, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
