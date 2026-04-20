# Haddaf ⚽ — Football Action Recognition

An AI-powered system that analyzes football video footage and recognizes player actions 


## ⚠️ Important: Before You Start

The `models/` folder is **not included** in this repository because the files are too large for GitHub.

**You must get the `models/` folder separately** (ask the team leader to share it via Google Drive or OneDrive), then place it inside the project folder like this:

```
Haddaf_model-main/
├── models/
│   ├── best_ensemble_classifier.pkl
│   ├── feature_scaler_lgbm.joblib
│   └── label_encoder_lgbm.joblib
├── server.py
├── test_server.py
└── ...
```

---

## 🚀 How to Run

### 1. Install Requirements

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Set Your Video Path

Open `test_server.py` and update the video path to point to your video file:

```python
# Change this to your actual video path
video_path = "C:/Users/yourname/path/to/your/video.mp4"
```

### 3. Start the Server

In one terminal, run:

```bash
python server.py
```

Wait until you see the server is running.

### 4. Run the Test

In a **second terminal**, run:

```bash
python test_server.py
```

---

## 📁 Project Structure

```
├── models/              ← AI models (get separately, not on GitHub)
├── trackers/            ← Player tracking logic
├── utils/               ← Helper functions
├── server.py            ← Main server (run this first)
├── test_server.py       ← Send video and get results (run this second)
├── main.py              ← Core processing logic
├── action_recognizer.py ← Action recognition module
├── requirements.txt     ← Python dependencies
└── Dockerfile           ← Docker configuration
```

---

## 🛠️ Requirements

- Python 3.x
- All packages listed in `requirements.txt`
- The `models/` folder (shared separately)
