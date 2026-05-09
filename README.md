# Pokémon Cry Recognition System

A machine learning-based Pokémon cry recognition system built with Python.

This project recognizes Pokémon from audio input using MFCC feature extraction and a RandomForest classifier, with a desktop GUI that supports:

* Audio file selection
* Microphone recording
* Top-3 prediction
* Confidence thresholding
* Pokémon image display
* Multilingual Pokémon names

The project was originally inspired by the idea of identifying Pokémon encounter sounds in arcade environments.

---

# Overview

The system takes a Pokémon cry audio file or microphone recording as input, extracts audio features, predicts the most likely Pokémon, and displays the result through a GUI.

Main workflow:

```text
Audio Input
↓
Preprocessing
↓
MFCC Feature Extraction
↓
RandomForest Classifier
↓
Top-3 Prediction
↓
GUI Result Display
```

---

# Features

* Recognition support for approximately 1000+ Pokémon
* Audio upload support
* Microphone recording support
* MFCC-based audio feature extraction
* RandomForest-based classification
* Top-3 prediction display
* Confidence threshold / Unknown detection
* Pokémon image display
* Automatic Pokémon image downloading
* English / Japanese / Chinese Pokémon names
* Audio augmentation pipeline
* Dataset generation pipeline
* Model evaluation tools
* Spectrogram generation pipeline
* Experimental CNN training pipeline

---

# Current Performance

Evaluation on the clean and augmented dataset:

| Metric         | Accuracy |
| -------------- | -------: |
| Top-1 Accuracy |   99.06% |
| Top-3 Accuracy |   99.70% |
| Top-5 Accuracy |   99.82% |

Note:

These results are evaluated on the generated clean/augmented dataset and do not fully represent real-world arcade or microphone recording performance.

---

# Model

## Main Model

The current main model uses:

```text
MFCC + RandomForestClassifier
```

This approach performed well under limited per-class data conditions.

## Experimental CNN

An experimental CNN model using Mel Spectrogram images is also included.

Current observations suggest that:

```text
MFCC + RandomForest significantly outperformed
the baseline CNN under limited per-class data conditions.
```

---

# Project Structure

```text
pokemon-cry-recognition/
│
├── src/
│   ├── app.py
│   ├── build_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate_model.py
│   ├── generate_spectrograms.py
│   ├── train_cnn.py
│   └── download_images.py
│
├── models/
│   └── pokemon_names.json
│
├── cries/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/pokemon-cry-recognition.git
cd pokemon-cry-recognition
```

---

## 2. Create Virtual Environment

```bash
py -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

---

## 3. Install Dependencies

```bash
py -m pip install -r requirements.txt
```

---

## 4. Install FFmpeg

This project requires FFmpeg for audio processing.

Download FFmpeg from:

https://ffmpeg.org/download.html

Or install on Windows using:

```powershell
winget install ffmpeg
```

Verify installation:

```powershell
ffmpeg -version
```

---

# Dataset Generation

The dataset generation script:

* Reads Pokémon cry audio files
* Retrieves multilingual Pokémon names
* Creates ID-based folders
* Applies audio augmentation
* Extracts MFCC features
* Saves training data

Run:

```bash
py src/build_dataset.py
```

Generated files:

```text
models/features.npy
models/labels.npy
models/pokemon_names.json
```

---

# Training the Main Model

```bash
py src/train_model.py
```

This trains the RandomForest classifier and generates:

```text
models/pokemon_cry_model.pkl
models/label_encoder.pkl
```

Note:

The trained model files are intentionally not included in this repository because of GitHub file size limitations.

Users should generate the model locally.

---

# Running the GUI

```bash
py src/app.py
```

The GUI supports:

```text
Select Audio File
Record Audio
Play Audio
Predict
Top-3 Result Display
Pokémon Image Display
```

---

# Command-Line Prediction

Example:

```bash
py src/predict.py data_raw/025_Pikachu/25.ogg
```

Example output:

```text
Predicted Pokémon: Pikachu
Confidence: 86.67%
```

---

# Model Evaluation

```bash
py src/evaluate_model.py
```

Generated evaluation outputs:

```text
evaluation_results/
├── classification_report.csv
├── confusion_matrix.csv
├── sample_predictions.csv
├── weak_classes_by_recall.csv
└── low_confidence_predictions.csv
```

The evaluation script also reports:

```text
Top-1 Accuracy
Top-3 Accuracy
Top-5 Accuracy
```

---

# Spectrogram Generation

To generate Mel Spectrogram images:

```bash
py src/generate_spectrograms.py
```

Generated output:

```text
spectrograms/
```

These images are used for the experimental CNN model.

---

# CNN Experiment

Train the experimental CNN model:

```bash
py src/train_cnn.py
```

Current observation:

```text
MFCC + RandomForest performs significantly better
than the baseline CNN under limited per-class data conditions.
```

Future improvements may include:

* More augmented samples
* Better CNN architectures
* Transfer learning
* Real-world recording datasets
* Noise simulation

---

# Real-World Usage Goal

The long-term goal of this project is to build a Pokémon arcade encounter sound assistant.

Target workflow:

```text
Arcade encounter sound
↓
Microphone recording
↓
Prediction
↓
Top-3 Pokémon candidates
```

Because real arcade environments contain:

* Background noise
* Speaker distortion
* Reverberation
* Human voices

future development will focus on noise robustness and real-world adaptation.

---

# Future Work

* Real arcade recording dataset
* Better microphone preprocessing
* Silence trimming
* Volume normalization
* Background noise augmentation
* Arcade speaker simulation
* Improved Unknown detection
* Better CNN performance
* Mobile application version
* Real-time continuous listening mode
* Executable packaging

---

# Technologies Used

* Python
* Librosa
* NumPy
* Scikit-learn
* RandomForestClassifier
* CustomTkinter
* Pillow
* SoundDevice
* SoundFile
* PyTorch
* Matplotlib

---

# Disclaimer

Pokémon is a trademark of Nintendo, Game Freak, Creatures, and The Pokémon Company.

This project is a non-commercial educational and experimental project.

It is not affiliated with or endorsed by Nintendo, Game Freak, Creatures, or The Pokémon Company.