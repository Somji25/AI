import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


model = load_model("/AI_python.h5")


classes = ['พ่อ', 'ดีใจ', 'มีความสุข', 'ชอบ', 'ไม่สบาย', 'เข้าใจแล้ว', 'เศร้า', 'ยิ้ม', 'โชคดี', 'หิว', 'ชอบ',
           'แม่', 'ขอความช่วยเหลือ', 'ฉัน', 'เขา', 'ขอโทษ', 'ขอโทษ', 'เป็นห่วง', 'เป็นห่วง', 'รัก', 'เขา',
           'สวัสดี', 'แม่', 'สวัสดี', 'เสียใจ', 'เสียใจ', 'ขอบคุณ', 'ยิ้ม', 'อิ่ม', 'แม่', 'รัก',
           'รัก', 'เข้าใจแล้ว', 'เข้าใจแล้ว', 'ขอความช่วยเหลือ', 'ห', 'ฬ', 'อ', 'ฮ']

# โหลดฟอนต์ภาษาไทย
fontpath = "/Bethai.ttf"
font = ImageFont.truetype(fontpath, 30)

async def process_video(websocket, path):
    cap = cv2.VideoCapture(0)  # เปิดกล้อง
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            # เปลี่ยนเป็นสี RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # ตรวจจับมือในภาพ
            results = hands.process(image)

            # เปลี่ยนกลับเป็น BGR เพื่อแสดงผล
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                all_x_min, all_y_min, all_x_max, all_y_max = [], [], [], []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = image.shape
                    x_min = int(min([landmark.x * w for landmark in hand_landmarks.landmark]))
                    y_min = int(min([landmark.y * h for landmark in hand_landmarks.landmark]))
                    x_max = int(max([landmark.x * w for landmark in hand_landmarks.landmark]))
                    y_max = int(max([landmark.y * h for landmark in hand_landmarks.landmark]))

                    all_x_min.append(x_min)
                    all_y_min.append(y_min)
                    all_x_max.append(x_max)
                    all_y_max.append(y_max)

                # คำนวณกรอบรอบทั้งสองมือ
                x_min = max(0, min(all_x_min) - 20)
                y_min = max(0, min(all_y_min) - 20)
                x_max = min(w, max(all_x_max) + 20)
                y_max = min(h, max(all_y_max) + 20)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (102, 0, 0), 2)

                hand_roi = image[y_min:y_max, x_min:x_max]

                if hand_roi.size > 0:
                    hand_roi_resized = cv2.resize(hand_roi, (148, 148))
                    hand_roi_resized = np.array(hand_roi_resized, dtype="float32") / 255.0
                    hand_roi_resized = np.expand_dims(hand_roi_resized, axis=0)

                    try:
                        predictions = model.predict(hand_roi_resized)
                        predicted_class = np.argmax(predictions[0])
                        predicted_letter = classes[predicted_class]
                        confidence = predictions[0][predicted_class] * 100

                        # ใช้ PIL เพื่อสร้างภาพข้อความ
                        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(image_pil)

                        # ข้อความที่จะแสดง
                        text_th = f'ทำนาย: {predicted_letter} {confidence:.1f}%'

                        # หาขนาดข้อความ
                        bbox = draw.textbbox((0, 0), text_th, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        # วาดกรอบพื้นหลังสำหรับข้อความ
                        draw.rectangle([x_min, y_min - text_height - 10, x_min + text_width, y_min], fill=(0, 0, 102))
                        # วาดข้อความ
                        draw.text((x_min + 5, y_min - text_height - 5), text_th, font=font, fill=(255, 255, 255))

                        # แปลงภาพกลับไปยัง OpenCV
                        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                    except Exception as e:
                        print(f"Error in prediction: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            await websocket.send(buffer.tobytes())

# เริ่ม WebSocket server
async def main():
    async with websockets.serve(process_video, "localhost", 8765):
        await asyncio.Future()  # ทำให้รันเซิร์ฟเวอร์ตลอดเวลา

if __name__ == "__main__":
    asyncio.run(main())
