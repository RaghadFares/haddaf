# Haddaf — Football Action Recognition

An AI-powered system that analyzes football video footage and recognizes player actions.

---

## Important: Before You Start

The `models/` folder is not included in this repository due to file size limitations.

You must download it separately from the link below and place it inside the project folder before running anything.

**Download models folder:** https://drive.google.com/drive/folders/1f2kR-Oso6yrBaIdugcLPEWAR4mmOmDRh?usp=sharing

After downloading, your project structure should look like this:

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

## How to Run

**Step 1 — Install requirements**

```bash
pip install -r requirements.txt
```

**Step 2 — Set your video path**

Open `test_server.py` and update the video path to point to your video file:

```python
video_path = "C:/Users/yourname/path/to/your/video.mp4"
```

**Step 3 — Start the server**

In one terminal, run:

```bash
python server.py
```

Wait until the server is running before moving to the next step.

**Step 4 — Run the test**

In a second terminal, run:

```bash
python test_server.py
```

---

## Project Structure

```
├── models/              # AI models (download separately)
├── trackers/            # Player tracking logic
├── utils/               # Helper functions
├── server.py            # Main server
├── test_server.py       # Sends video and retrieves results
├── main.py              # Core processing logic
├── action_recognizer.py # Action recognition module
├── requirements.txt     # Python dependencies
└── Dockerfile           # Docker configuration
```

---

## Requirements

- Python 3.x
- All packages listed in `requirements.txt`
- The `models/` folder downloaded from the link above
