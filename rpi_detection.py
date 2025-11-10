import os
import time
import json
import requests
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from ultralytics import YOLO

# --- Load Environment Variables ---
load_dotenv()

# --- API Key Configuration ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"⚠️ Could not configure Gemini API. Error: {e}")
    gemini_model = None

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ Telegram credentials missing in .env — alerts disabled.")

# --- Constants ---
YOLO_MODEL_PATH = "fall_yolov8n_35epochs.pt"
CONFIDENCE_THRESHOLD = 0.80  # For alerts
DISPLAY_CONFIDENCE_THRESHOLD = 0.30  # Lower threshold for displaying boxes
FALL_CLASS_ID = 0
COOLDOWN_PERIOD_SECONDS = 20

LLM_PROMPT = """
You are an AI vision system designed to assist emergency responders. Analyze the provided image and respond ONLY in valid JSON format with the following keys:

- "fall_detected": boolean,
- "context": string (2-line description of the person and surroundings),
- "bleeding_observed": boolean,
- "person_condition": string ("alert", "unresponsive", "injured", "bleeding", "unknown"),
- "confidence": float (0-100)
"""

# --- Load YOLO Model ---
try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO model loaded successfully: {YOLO_MODEL_PATH}")
    print("📋 Model class names:", model.names)
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    exit(1)

# --- YOLO Inference ---
def run_yolo_inference(frame):
    try:
        results = model(frame, verbose=False)[0]
        detections = []
        
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Always add to detections for display, but mark if it meets alert threshold
                meets_alert_threshold = conf >= CONFIDENCE_THRESHOLD
                detections.append((x1, y1, x2, y2, conf, cls, meets_alert_threshold))
                
                print(f"   Box {i}: {class_name} conf={conf:.3f} {'✅' if meets_alert_threshold else '⚠️'}")
                
        else:
            print("   No boxes detected in this frame")
            
        return detections
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return []

# --- LLM Verification ---
def verify_fall_with_llm(frame, prompt):
    if not gemini_model:
        print("   LLM not configured. Skipping verification.")
        return None

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        print("   ⏳ Sending frame to Gemini...")
        response = gemini_model.generate_content([prompt, pil_image])
        json_text = response.text.strip()
        if json_text.startswith('```json'):
            json_text = json_text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return None

# --- Telegram ---
def send_telegram_alert(llm_response):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("   ❌ Telegram disabled.")
        return

    try:
        status = llm_response.get('person_condition', 'unknown').upper()
        confidence = llm_response.get('confidence', 0.0)
        context = llm_response.get('context', 'No context.')
        bleeding = "Yes" if llm_response.get('bleeding_observed', False) else "No"

        message = (
            f"🚨 *HIGH CONFIDENCE FALL DETECTED* 🚨\n\n"
            f"*Status:* {status}\n"
            f"*Confidence:* {confidence:.2f}%\n"
            f"*Bleeding Observed:* {bleeding}\n\n"
            f"*Context:*\n{context}"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(url, data=payload, timeout=5)
        if response.status_code == 200:
            print("   ✅ Telegram alert sent.")
        else:
            print(f"   ❌ Telegram failed: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"   ❌ Telegram error: {e}")

# --- Main ---
def main():
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("✅ Pi Camera started.")

    cv2.namedWindow("Live Fall Detection", cv2.WINDOW_NORMAL)
    last_llm_call_time = 0
    frame_count = 0
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue

            # Convert 4-channel BGRA or RGB to 3-channel BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            display_frame = frame.copy()
            current_time = time.time()
            frame_count += 1
            fps_frame_count += 1

            # Calculate FPS every second
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (current_time - fps_start_time)
                fps_start_time = current_time
                fps_frame_count = 0

            # Run inference every 10 frames to reduce CPU load
            if frame_count % 10 == 1:
                print(f"\n--- Frame {frame_count} ---")
                detections = run_yolo_inference(frame)
            else:
                detections = []

            # --- Drawing logic ---
            yolo_triggered = False

            for (x1, y1, x2, y2, conf, cls_id, meets_alert_threshold) in detections:
                # Get class name
                class_name = model.names.get(cls_id, f"Class{cls_id}")
                
                # Choose color based on confidence
                if meets_alert_threshold:
                    color = (0, 0, 255)  # Red for high confidence
                    label = f"FALL: {conf:.2f}"
                    yolo_triggered = True
                elif conf >= DISPLAY_CONFIDENCE_THRESHOLD:
                    color = (0, 165, 255)  # Orange for medium confidence
                    label = f"FALL?: {conf:.2f}"
                else:
                    color = (0, 255, 0)  # Green for low confidence
                    label = f"FALL??: {conf:.2f}"

                # Draw the rectangle and text for ALL detections above display threshold
                if conf >= DISPLAY_CONFIDENCE_THRESHOLD:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add frame counter and status to display
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detections: {len(detections)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Alert Ready: {yolo_triggered}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Only trigger LLM for high confidence detections
            if yolo_triggered and (current_time - last_llm_call_time >= COOLDOWN_PERIOD_SECONDS):
                print("⚠️ High-confidence fall detected → LLM verification...")

                if gemini_model:
                    llm_start = time.time()
                    result = verify_fall_with_llm(frame, LLM_PROMPT)
                    duration = time.time() - llm_start

                    if result:
                        last_llm_call_time = current_time
                        is_fall = result.get('fall_detected', False)
                        conf = result.get('confidence', 0.0)
                        if is_fall and conf >= 80.0:
                            print("🚨 FALL CONFIRMED by LLM!")
                            send_telegram_alert(result)
                        else:
                            print(f"✅ LLM rejected fall (confidence: {conf:.1f}%)")

            cv2.imshow("Live Fall Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("👋 Done.")

if __name__ == "__main__":
    main()