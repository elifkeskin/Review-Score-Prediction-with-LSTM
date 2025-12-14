# LSTM Review Score Prediction

A Python project that uses an LSTM (Long Short‑Term Memory) neural network to predict review scores based on textual comments.  
The repository contains the model implementation, data preprocessing scripts, and utilities for training and inference.

## Features
- End‑to‑end LSTM regression model
- Data preprocessing and tokenization
- Prediction script for new reviews
- Simple command‑line interface

## Requirements
See `requirements.txt` for the full list of dependencies.  
Typical packages include:
- `numpy`
- `pandas`
- `tensorflow` (or `torch` if you adapt the code)
- `scikit-learn`

## Usage

### Training
```bash
python lstm_regression.py
```

### Predicting a review score
```bash
python predict_review.py <path_to_review_text_file>
```

## Project Structure
LSTM_ile_Yorum_Puan_Tahmini/
├─ lstm_regression.py      # Model definition and training
├─ predict_review.py       # Inference script
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation (this file)
└─ LICENSE                 # License information