# Cognitive_Assessment-and-Assistance-using-DL-and-LLMs
A Multimodal AI System for Early Detection, Tracking, and Assistance in Cognitive Decline

This repository provides an end-to-end framework for assessing cognitive impairment (such as dementia) using deep learning, speech transcription, and Retrieval-Augmented Generation (RAG) systems. Tailored for scalability and accessibility, especially in low-resource settings, it integrates screening, personalized assistance, and explainable analytics to enhance patient and clinician outcomes.

## Directory Structure

```
.
├── model/
│   ├── api_net.py                  # Clock Drawing Test evaluator (API-Net model)
│   ├── clock_analyser.py          # Evaluates hand-drawn clock images
│   ├── audio_transcription.py     # Whisper-based transcription of patient speech
│   ├── Alphabet_Distortion.py     # Detects alphabet distortion in patient input
│   ├── cube_analyser.py           # Analysis of cube-drawing task
│   ├── double_infinity_analyser.py# Handles analysis of double-infinity shape drawings
│   ├── word_validation.py         # Validates animal-naming test results
│   └── animal_names.py            # Word fluency assessment
│
├── frontend-html/
│   ├── index.html                 # Entry point for cognitive testing
│   ├── home.html                  # Home UI for patients or clinicians
│   ├── fluency.html               # Frontend for word-naming task
│   └── visuospatial.html          # UI for drawing tasks (clock, cube, etc.)
│
├── runner.py                      # Unified entry script for running assessments
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .gitignore

```

## Project Objectives

- Enable multimodal cognitive evaluation (speech, image, verbal)

- Improve accessibility via digital tools

- Deploy XAI dashboards (future)

- Offer real-time assistance via LLMs and RAG agents (future)

## Key Features 

Multimodal Cognitive Testing: Combines analysis of visuospatial drawings (clock, cube), speech, and word recall.

Speech Evaluation: Uses Whisper + NLP pipelines to analyze fluency, coherence, and early dementia indicators.

Deep Learning with API-Net: Fine-grained scoring of clock drawings using an attention-based pairwise comparison network.

Longitudinal Tracking (Planned): Will integrate scoring history for each patient across sessions.

Assistive Feedback (Planned): Supports integration with RAG-based agents for memory support.

## Setup Guide

### 1. **Clone the Repository**

```bash
git clone https://github.com/iamnithyaramesh/Cognitive_Assessment-and-Assistance-using-DL-and-LLMs.git
cd Cognitive_Assessment-and-Assistance-using-DL-and-LLMs
```

---

### 2. **Steup the virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. **Model Setup for LLM-based Evaluation**
```bash
python runner.py
```

---

## Modules Overview

| Module                                             | Description                                                     |
| -------------------------------------------------- | --------------------------------------------------------------- |
| `api_net.py`                                       | Loads and applies the API-Net model for scoring clock drawings. |
| `clock_analyser.py`                                | Interface to preprocess clock images and apply scoring logic.   |
| `audio_transcription.py`                           | Transcribes speech using Whisper and applies LLM-based scoring. |
| `Alphabet_Distortion.py`                           | Checks for distortions in drawn alphabets, a potential marker.  |
| `animal_names.py` & `word_validation.py`           | Used for verbal fluency analysis (animal naming task).          |
| `cube_analyser.py` & `double_infinity_analyser.py` | Process and score other ACE-III-inspired drawing tasks.         |

---

## Front-end Interface 

These static files can be served via Flask or any local web server.

- index.html: Start screen

- home.html: Home interface

- fluency.html: Interface for fluency scoring modules

- visuospatial.html: Interface for visuospatial scoring modules.

You can serve this folder using a simple server like:

```bash
cd frontend-html
python -m http.server 8080
```
Then open http://localhost:8080 in your browser.

## Dataset Used
Clock Drawing - Visuospatial Section: Schulman Clock Drawing images from MoCA/ACE-III datasets

GitHub: https://github.com/cccnlab/CDT-API-Network







