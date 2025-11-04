import google.generativeai as genai
import os
import time
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# --- API Key Configuration ---
from dotenv import load_dotenv

# Load environment variables from .env file
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
    print("   Please set the GEMINI_API_KEY environment variable in your .env file and restart the script.")
    gemini_model = None  # Set model to None so the script can run but skip LLM calls

# --- Constants ---
YOLO_MODEL_PATH = "fall_yolov8n_35epochs.pt"
WEBCAM_INDEX = 0

# Fall Detection Thresholds
CONFIDENCE_THRESHOLD = 0.80  # YOLO confidence score to trigger LLM verification
FALL_CLASS_ID = 0            # Based on the training output {0: 'Fall-Detected'}

# LLM Cooldown
COOLDOWN_PERIOD_SECONDS = 20 # Cooldown to prevent spamming the LLM API

# --- LLM Prompt ---
LLM_PROMPT = """
You are an AI vision system designed to assist emergency responders. Analyze the provided image and respond ONLY in valid JSON format with the following keys:

- "fall_detected": boolean,
- "context": string (must be a 2 line description of the condition of the person and the surroundings; estimate the cause of fall as well),
- "bleeding_observed": boolean,
- "person_condition": string (one of: "alert", "unresponsive", "injured", "bleeding", "unknown"),
- "confidence": float (a score in percentage (0-100)% how sure are you about your prediction in percentage)

Do not include any other text.
"""

# --- Load YOLO Model ---
try:
    fall_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLOv8 Model loaded from: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    print("   Make sure {YOLO_MODEL_PATH} is correct.")
    fall_model = None

# --- LLM Verification Function ---
def verify_fall_with_llm(frame, prompt):
    """
    Sends a frame to the Gemini LLM for high-confidence fall verification.
    """
    if not gemini_model:
        print("LLM is not configured. Skipping verification.")
        return None

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        print(f"   ⏳ Sending frame to Gemini... (This will block until response is received)")
        response = gemini_model.generate_content([prompt, pil_image])

        json_text = response.text.strip()

        # Robustly strip markdown fences
        if json_text.startswith('```json'):
            json_text = json_text.strip().replace('```json', '').replace('```', '').strip()

        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print(f"❌ LLM JSON Decode Error: {e}")
        print(f"   LLM Raw Response: {response.text}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during LLM call (Possible API Timeout): {e}")
        return None

# --- Main Execution Block ---
def main():
    if not fall_model:
        print("❌ YOLO model is not loaded. Cannot start processing.")
        return

    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"❌ Error opening webcam (Index: {WEBCAM_INDEX}).")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Webcam Info: {frame_width}x{frame_height} @ {fps:.2f} FPS.")
    print("Press 'q' to quit.")

    cv2.namedWindow("Live Fall Detection", cv2.WINDOW_NORMAL)

    last_llm_call_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Webcam feed ended.")
            break

        current_time = time.time()
        frame_count += 1
        display_frame = frame.copy()  # Create a copy for drawing

        # --- STAGE 1: Real-Time Edge Detection (YOLOv8) ---
        yolo_triggered = False
        results = fall_model(frame, verbose=False)

        if results and results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls[0]) == FALL_CLASS_ID:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Check if this detection triggers the LLM
                    if conf >= CONFIDENCE_THRESHOLD:
                        yolo_triggered = True
                        color = (0, 0, 255)  # Red for high confidence
                        label = f"FALL DETECTED: {conf:.2f}"
                    else:
                        color = (0, 255, 0)  # Green for low confidence
                        label = f"Fall: {conf:.2f}"
                        
                    # Draw rectangle and text
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # If the YOLO confidence threshold is met:
        if yolo_triggered:
            print(f"\n--- Frame {frame_count} ---")
            print(f"⚠️ YOLO Fall Detection > {CONFIDENCE_THRESHOLD*100:.2f}%! Initiating LLM Verification.")

            # --- STAGE 2: Multimodal LLM Verification (Gemini) ---
            if current_time - last_llm_call_time >= COOLDOWN_PERIOD_SECONDS:
                if not gemini_model:
                    print("   Skipping LLM call: API not configured.")
                else:
                    llm_start_time = time.time()
                    verification_result = verify_fall_with_llm(frame, LLM_PROMPT)
                    llm_duration = time.time() - llm_start_time

                    if verification_result:
                        last_llm_call_time = current_time  # Update cooldown timer

                        is_fall = verification_result.get('fall_detected', False)
                        confidence = verification_result.get('confidence', 0.0)
                        condition = verification_result.get('person_condition', 'unknown')
                        context = verification_result.get('context', 'No context provided.')

                        if is_fall and confidence >= 80.0:
                            print("🚨 HIGH CONFIDENCE FALL ALERT!")
                            print(f"   Status: {condition.upper()} | Confidence: {confidence:.2f}% | LLM Time: {llm_duration:.2f}s")
                            print(f"   Context: {context}")
                            print("----------------------------------------------------------------")
                        else:
                            print(f"✅ LLM Verification: Fall not confirmed or confidence low ({confidence:.2f}%).")
                            print(f"   LLM Time: {llm_duration:.2f}s. Context: {context}")

                    else:
                        print(f"LLM call failed or returned unparseable data. Duration: {llm_duration:.2f}s")
            
            else:
                cooldown_remaining = COOLDOWN_PERIOD_SECONDS - (current_time - last_llm_call_time)
                print(f"   ⏳ Cooldown in effect. Skipping LLM call. Remaining: {cooldown_remaining:.2f}s")

        # --- Display the Frame ---
        cv2.imshow("Live Fall Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nLive processing stopped. 🎬")

if __name__ == "__main__":
    main()
