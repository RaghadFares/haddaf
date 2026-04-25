# Haddaf — AI-Powered Football Action Recognition Engine

## What is Haddaf?

**Haddaf** is an AI-Powered Football Talents Discovery and Tracking Application.

The goal of the project is to provide players, coaches, and scouts with an intelligent system to evaluate player performance based on match video clips. By applying computer vision and machine learning techniques, Haddaf automatically extracts key performance metrics — giving players a platform to showcase their skills, helping coaches track their players' performance, and supporting scouts in talent discovery.

---

## This Repository — The Brain of Haddaf

This repository contains the **core AI engine** that powers the Haddaf application. It is the heart of the system — responsible for processing football match video clips and automatically recognizing the actions performed by a specific player.

When the Haddaf mobile application sends a video clip of a player, this engine:
1. Tracks the selected player throughout the video
2. Extracts their body pose frame by frame
3. Classifies what action they are performing (dribble, pass, shoot, header, or tackle)
4. Returns the action counts back to the application

Without this engine, the Haddaf app cannot analyze or evaluate player performance.

---

## Important: Before You Start

The `models/` folder is not included in this repository due to file size limitations.
You must download it separately from the link below and place it inside the project folder before running anything.

**Download models folder:**
[https://drive.google.com/drive/folders/1f2kR-Oso6yrBaIdugcLPEWAR4mmOmDRh?usp=sharing](https://drive.google.com/drive/folders/1f2kR-Oso6yrBaIdugcLPEWAR4mmOmDRh?usp=sharing)

**Download testing videos and coordinates PDF:**
[https://drive.google.com/drive/u/2/folders/1_55ijHhFhqmOD9QFI4GcJ7QteenO6LfG](https://drive.google.com/drive/u/2/folders/1_55ijHhFhqmOD9QFI4GcJ7QteenO6LfG)

The testing videos folder also contains a **PDF file** that includes the **coordinates required for each test video**. You must use the correct coordinates that correspond to the selected video.

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

### Step 1 — Install requirements

```bash
pip install -r requirements.txt
```

---

### Step 2 — Choose a Testing Video

1. Download the testing videos from:
[https://drive.google.com/drive/u/2/folders/1_55ijHhFhqmOD9QFI4GcJ7QteenO6LfG](https://drive.google.com/drive/u/2/folders/1_55ijHhFhqmOD9QFI4GcJ7QteenO6LfG)
2. Open the **PDF file inside the testing videos folder**.
3. Find the coordinates that correspond to your selected video.

---

### Step 3 — Set Video Path and Coordinates

Open `test_server.py`.

Update the **video path** to your selected testing video:

```python
video_path = "C:/Users/yourname/path/to/your/video.mp4"
```

Then update the **coordinates** using the values provided in the **PDF** for that specific video.

Make sure:
- The video path matches your local file location
- The coordinates match the selected video
- The correct coordinate set is used for accurate action detection

---

### Step 4 — Start the Server

In one terminal, run:

```bash
python server.py
```

Wait until the server is fully running before moving to the next step.

---

### Step 5 — Run the Test

In a second terminal, run:

```bash
python test_server.py
```

The system will:
- Send the video to the server
- Process player movements
- Detect football actions
- Return the recognized results

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
- The `models/` folder downloaded from the models link
- Testing videos downloaded from the testing videos link
- Coordinates selected correctly from the PDF inside the testing videos folder
