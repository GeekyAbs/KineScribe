# KineScribe: Hybrid Edge-LLM Framework for Fall Detection

> **⚠️ Work in Progress**: This project is currently under active development.

KineScribe is a robust fall detection system that combines the speed of on-device edge computing with the analytical power of multimodal Large Language Models (LLMs). This hybrid architecture provides rapid initial detection at the edge while leveraging cloud-based AI to eliminate false positives and generate context-rich, human-readable alerts for informed emergency response.

## 🎯 Key Features

- **Real-time edge detection** using YOLOv8 pose estimation on Raspberry Pi
- **AI-powered verification** via multimodal LLM (Google Gemini) to reduce false positives
- **Context-aware alerting** with structured JSON output for emergency responders
- **High accuracy**: 85.25% mAP50, 84.90% precision on test set

## 🏗️ Architecture Overview

The detection and alert process consists of four key stages:

### 1. Real-Time Edge Detection (Raspberry Pi)

A lightweight YOLOv8 pose estimation model runs continuously on a Raspberry Pi, analyzing video feeds in real-time.

- **Low-Latency Analysis**: Immediate pose data extraction from video feed
- **Smart Triggering**: Detects fall-indicative postures using keypoint heuristics
- **Frame Capture**: Captures and forwards suspicious frames for verification

### 2. Multimodal LLM Verification (GenAI)

Potential fall frames are sent to a multimodal LLM (e.g., Google Gemini) for advanced analysis.

- **Contextual Understanding**: Evaluates both posture and surrounding scene context
- **False Positive Reduction**: Distinguishes between actual falls and benign situations (e.g., lying on a bed vs. fallen on the floor)
- **Intelligent Decision-Making**: Provides high-confidence fall confirmation

### 3. Context-Rich Alerting

Upon LLM confirmation, the system generates human-readable, structured alerts with actionable information.

### 4. LLM Inference and Alert Format

The system outputs structured JSON alerts that can be easily parsed by downstream applications.

**Example Alert Output:**
```json
{
  "fall_detected": true,
  "context": "A person is lying on their back on the carpeted floor of an office, with knees bent and arms near their head.\nThey appear unresponsive; the cause of the fall is estimated to be a medical event or accidental trip.",
  "bleeding_observed": false,
  "person_condition": "unresponsive",
  "confidence": 90.0
}
```

## 📊 Performance Metrics

The YOLOv8 edge model was trained on a pseudo-labeled dataset and achieved the following results on the test set:

| Metric | Value |
|--------|-------|
| **mAP50** | 85.25% |
| **mAP50-95** | 51.32% |
| **Precision** | 84.90% |
| **Recall** | 77.48% |
| **F1 Score (fall class)** | 81.02% |

## 📂 Repository Structure
```
.
├── FallDetection_ModelTraining.ipynb      # YOLOv8 edge model training
├── FallDetection_PseudoLabelling.ipynb    # Pseudo-label generation from COCO dataset
└── FallDetection_LLM_Inference.ipynb      # LLM verification demo with structured output
```

### Notebooks

- **`FallDetection_ModelTraining.ipynb`**: Complete training pipeline for the YOLOv8 fall detection model deployed on edge devices
- **`FallDetection_PseudoLabelling.ipynb`**: Automated pseudo-labeling process using YOLOv8-pose on COCO dataset to generate high-quality training data
- **`FallDetection_LLM_Inference.ipynb`**: Demonstration of multimodal LLM integration for fall verification with structured JSON response

## 🚀 Getting Started

### Prerequisites

- Raspberry Pi (for edge deployment)
- Python 3.8+
- Google Gemini API access (or compatible multimodal LLM)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/kinescribe.git
cd kinescribe

```

### Usage

1. **Train the Edge Model**: Run `FallDetection_ModelTraining.ipynb` to train or fine-tune the YOLOv8 model
2. **Generate Pseudo-Labels** (optional): Use `FallDetection_PseudoLabelling.ipynb` to create additional training data
3. **Test LLM Integration**: Explore `FallDetection_LLM_Inference.ipynb` to understand the verification approach

### Run the KineScribe Frontend

KineScribe ships with a Streamlit frontend for quick testing.

1) Install dependencies (Python 3.9–3.11 recommended):
```bash
pip install -r requirements.txt
```

2) Ensure your trained YOLO weights file exists (default expected at `fall_yolov8n_35epochs.pt`).

3) Provide credentials (either set environment variables or paste in the UI):
```bash
set GEMINI_API_KEY=your_gemini_key              # Windows PowerShell: $env:GEMINI_API_KEY="..."
set TELEGRAM_BOT_TOKEN=your_bot_token          # optional if you plan to send alerts
set TELEGRAM_CHAT_ID=123456789                 # optional default chat id
```

4) Launch the app:
```bash
streamlit run kinescribe_app.py
```

5) In the browser UI:
- Upload a short test video (5–15s works best)
- Confirm the YOLO weights path (defaults to `fall_yolov8n_35epochs.pt`)
- Paste your Gemini API key
- Optionally paste your Telegram Bot Token and Chat/User ID to receive alerts
- Click “Run Detection” and monitor the live log

Notes:
- A Telegram alert is sent only when the LLM confirms a fall with ≥ 80% confidence.
- Cooldown between LLM calls is 10s to control API usage.

## 🔬 Technical Approach

### Edge Detection
- Utilizes YOLOv8 pose estimation for real-time keypoint detection
- Implements heuristic-based fall detection algorithms
- Optimized for low-power devices (Raspberry Pi)

### LLM Verification
- Leverages multimodal capabilities to analyze both pose and environment
- Custom prompting for structured JSON output
- Provides confidence scores and contextual descriptions


## 📧 Contact

Email: abeshchakraborty10@gmail.com

---

**Note**: This project is under active development. Features and documentation are subject to change.