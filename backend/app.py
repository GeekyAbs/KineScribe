import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import google.generativeai as genai
import requests
from dotenv import load_dotenv

# --- Config ---
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

YOLO_MODEL_PATH = "fall_yolov8n_35epochs.pt"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Constants
FALL_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_SECONDS = 20  # Not used in batch mode, but kept for reference

# --- Load Models ---
try:
    fall_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO model loaded.")
except Exception as e:
    fall_model = None
    print(f"❌ YOLO load failed: {e}")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini configured.")
else:
    gemini_model = None
    print("⚠️ GEMINI_API_KEY missing — LLM disabled.")

LLM_PROMPT = """
You are an AI vision system for emergency response. Analyze the image and respond ONLY in valid JSON:

{
  "fall_detected": boolean,
  "context": "2-line description of person + surroundings; estimate cause",
  "bleeding_observed": boolean,
  "person_condition": "alert|unresponsive|injured|bleeding|unknown",
  "confidence": 0.0  // percentage (0–100)
}
Do NOT add extra text or markdown.
""".strip()

# --- Utils ---
def verify_with_gemini(frame: np.ndarray):
    if not gemini_model:
        return {"error": "Gemini not configured"}
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        resp = gemini_model.generate_content([LLM_PROMPT, pil_img], stream=False)
        text = resp.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": f"LLM failed: {str(e)}", "raw": text if 'text' in locals() else ""}

def send_telegram_alert(chat_id: str, result: dict):
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return {"error": "Telegram config missing"}

    status = result.get("person_condition", "unknown").upper()
    conf = result.get("confidence", 0)
    context = result.get("context", "—")
    bleeding = "Yes" if result.get("bleeding_observed") else "No"

    msg = (
        f"🚨 *HIGH CONFIDENCE FALL DETECTED* 🚨\n\n"
        f"*Status:* {status}\n"
        f"*Confidence:* {conf:.1f}%\n"
        f"*Bleeding:* {bleeding}\n\n"
        f"*Context:*\n{context}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": f"Telegram failed: {e}"}

# --- Routes ---
@app.route("/")
def health():
    return jsonify({"status": "KineScribe Backend Running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if not fall_model:
        return jsonify({"error": "YOLO model not loaded"}), 500

    file = request.files.get("file")
    telegram_id = request.form.get("telegram_id", "").strip()
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    if not telegram_id:
        return jsonify({"error": "telegram_id required"}), 400

    # Save to temp file
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Handle image or video (first frame only for demo simplicity)
        results = []
        is_fall_triggered = False
        frame_to_analyze = None

        if suffix in [".jpg", ".jpeg", ".png"]:
            frame = cv2.imread(tmp_path)
            frame_to_analyze = frame.copy()
            yolo_res = fall_model(frame, verbose=False)
        elif suffix in [".mp4", ".avi", ".mov"]:
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()  # Only first frame for demo (extend later for full video)
            cap.release()
            if not ret:
                return jsonify({"error": "Could not read video"}), 400
            frame_to_analyze = frame.copy()
            yolo_res = fall_model(frame, verbose=False)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # YOLO stage
        yolo_detections = []
        if yolo_res and yolo_res[0].boxes:
            for box in yolo_res[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                yolo_detections.append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [round(x, 2) for x in xyxy]
                })
                if cls_id == FALL_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                    is_fall_triggered = True

        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename,
            "yolo_detections": yolo_detections,
            "yolo_triggered": is_fall_triggered,
            "gemini_result": None,
            "telegram_sent": False,
            "telegram_response": None
        }

        # LLM + Telegram stage (only if YOLO triggered)
        if is_fall_triggered:
            llm_res = verify_with_gemini(frame_to_analyze)
            response["gemini_result"] = llm_res

            if isinstance(llm_res, dict) and not llm_res.get("error"):
                fall_confirmed = llm_res.get("fall_detected") and llm_res.get("confidence", 0) >= 80.0
                if fall_confirmed:
                    tg_res = send_telegram_alert(telegram_id, llm_res)
                    response["telegram_sent"] = True
                    response["telegram_response"] = tg_res
                else:
                    response["telegram_sent"] = False
                    response["reason_skipped"] = "LLM confidence < 80% or fall not confirmed"
            else:
                response["error_llm"] = llm_res

        return jsonify(response)

    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)