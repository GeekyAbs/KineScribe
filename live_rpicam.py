import google.generativeai as genai
import os
import time
import json
import requests
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# ✅ Use tflite-runtime instead of full TensorFlow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("❌ tflite-runtime not installed. Run: pip install tflite-runtime")
    exit(1)

# --- Load Environment Variables ---
load_dotenv()

# --- API Key Configuration ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')  # ⚠️ 'gemini-2.5-flash' may not be public yet
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
YOLO_MODEL_PATH = "fall_yolov8n_35epochs_int8.tflite"  # quantized int8 model
CONFIDENCE_THRESHOLD = 0.80
FALL_CLASS_ID = 0
COOLDOWN_PERIOD_SECONDS = 20

LLM_PROMPT = """
You are an AI vision system designed to assist emergency responders. Analyze the provided image and respond ONLY in valid JSON format with the following keys:

- "fall_detected": boolean,
- "context": string (must be a 2 line description of the condition of the person and the surroundings; estimate the cause of fall as well),
- "bleeding_observed": boolean,
- "person_condition": string (one of: "alert", "unresponsive", "injured", "bleeding", "unknown"),
- "confidence": float (a score in percentage (0-100)% how sure are you about your prediction in percentage)

Do not include any other text.
"""

# --- Load TFLite Model ---
interpreter = None
input_details = output_details = None

try:
    interpreter = Interpreter(model_path=YOLO_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ TFLite model loaded: {YOLO_MODEL_PATH}")
    print(f"   Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"   Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")
except Exception as e:
    print(f"❌ Failed to load TFLite model: {e}")


# --- Inference Function (supports int8/float32 models) ---
def run_tflite_inference(frame):
    global interpreter, input_details, output_details
    if interpreter is None:
        return []

    input_shape = input_details[0]['shape']  # e.g. [1, 640, 640, 3]
    input_dtype = input_details[0]['dtype']
    _, model_h, model_w, _ = input_shape
    h_orig, w_orig = frame.shape[:2]

    # Preprocess: resize + type conversion
    resized = cv2.resize(frame, (model_w, model_h))
    
    # ✅ Handle quantization
    if input_dtype == np.uint8:
        # int8 quantized model: input [0,255] uint8
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    elif input_dtype == np.float32:
        # float model: usually [0.0, 1.0] or [-1,1]
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
    else:
        print(f"⚠️ Unexpected input dtype: {input_dtype}")
        return []

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # [1, N, 6] or [1, 84, 8400]

    detections = []

    # ✅ Handle common YOLOv8 TFLite formats:
    # Case A: Post-processed [1, N, 6] — x1,y1,x2,y2,conf,class
    if output_data.ndim == 3 and output_data.shape[-1] == 6:
        pred = output_data[0]  # [N, 6]
        for det in pred:
            x1, y1, x2, y2, conf, cls = det
            if conf < CONFIDENCE_THRESHOLD:
                continue
            # Scale to original image
            x1 = int(x1 * w_orig / model_w)
            y1 = int(y1 * h_orig / model_h)
            x2 = int(x2 * w_orig / model_w)
            y2 = int(y2 * h_orig / model_h)
            detections.append((x1, y1, x2, y2, float(conf), int(cls)))

    # Case B: Raw output [1, 84, 8400] — needs decoding (not implemented here)
    else:
        print(f"⚠️ Unsupported output shape {output_data.shape}. Expecting [1, N, 6].")
        # For full YOLO decoding (NMS etc.), consider using ultralytics' export with `nms=True`

    return detections


# --- LLM & Telegram (unchanged except Telegram URL fix) ---
def verify_fall_with_llm(frame, prompt):
    if not gemini_model:
        print("   LLM is not configured. Skipping verification.")
        return None

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        print(f"   ⏳ Sending frame to Gemini...")
        response = gemini_model.generate_content([prompt, pil_image], safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        })
        json_text = response.text.strip()
        if json_text.startswith('```json'):
            json_text = json_text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"❌ LLM JSON Decode Error: {e}")
        print(f"   LLM Raw Response: {response.text}")
        return None
    except Exception as e:
        print(f"❌ LLM call error: {e}")
        return None


def send_telegram_alert(llm_response):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("   ❌ Telegram disabled.")
        return

    try:
        status = llm_response.get('person_condition', 'unknown').upper()
        confidence = llm_response.get('confidence', 0.0)
        context = llm_response.get('context', 'No context.')
        bleeding = "Yes" if llm_response.get('bleeding_observed', False) else "No"

        message_text = (
            f"🚨 *HIGH CONFIDENCE FALL DETECTED* 🚨\n\n"
            f"*Status:* {status}\n"
            f"*Confidence:* {confidence:.2f}%\n"
            f"*Bleeding Observed:* {bleeding}\n\n"
            f"*Context:*\n{context}"
        )

        # 🔥 FIXED: Removed extra spaces in URL
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message_text,
            'parse_mode': 'Markdown'
        }

        response = requests.post(url, data=payload, timeout=5)
        if response.status_code == 200:
            print("   ✅ Telegram alert sent.")
        else:
            print(f"   ❌ Telegram failed: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"   ❌ Telegram error: {e}")


# --- Main ---
def main():
    # 🔥 FIXED: `fall_model` → `interpreter`
    if interpreter is None:
        print("❌ TFLite model not loaded. Exiting.")
        return

    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("✅ Pi Camera started.")

    cv2.namedWindow("Live Fall Detection", cv2.WINDOW_NORMAL)
    last_llm_call_time = 0
    frame_count = 0

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue

            # Convert to BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            display_frame = frame.copy()
            current_time = time.time()
            frame_count += 1

            # --- TFLite Inference ---
            yolo_triggered = False
            detections = run_tflite_inference(frame)

            for (x1, y1, x2, y2, conf, cls_id) in detections:
                if cls_id == FALL_CLASS_ID:
                    yolo_triggered = True
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"FALL: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- LLM + Telegram ---
            if yolo_triggered and (current_time - last_llm_call_time >= COOLDOWN_PERIOD_SECONDS):
                print(f"\n--- Frame {frame_count} ---")
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