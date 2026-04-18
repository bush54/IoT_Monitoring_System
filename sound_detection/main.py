import os
import numpy as np
import librosa
import joblib
import pandas as pd

def extract_features(file_path, sr=16000, duration=5):
    signal, sr = librosa.load(file_path, sr=sr, mono=True)
    target_length = sr * duration
    if len(signal) < target_length:
        padding = target_length - len(signal)
        signal = np.pad(signal, (0, padding))
    else:
        signal = signal[:target_length]

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    
    zcr = librosa.feature.zero_crossing_rate(signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    rms = librosa.feature.rms(y=signal)

    features = np.concatenate([
        mfcc_mean, mfcc_std, 
        [np.mean(zcr), np.std(zcr), np.mean(centroid), np.std(centroid),
         np.mean(bandwidth), np.std(bandwidth), np.mean(rolloff), np.std(rolloff),
         np.mean(rms), np.std(rms)]
    ])
    return features

# Load Trained Model and Scaler
try:
    loaded_model = joblib.load("models/audio_svm_model.pkl")
    loaded_scaler = joblib.load("models/scaler.pkl")
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run the training script first.")

# Detection Sample
test_file = "freesound_community-sawing-wood-96479.wav"

if os.path.exists(test_file):
    sample = extract_features(test_file).reshape(1, -1)
    sample = loaded_scaler.transform(sample)
    pred = loaded_model.predict(sample)
    print(f"Detection Result: The predicted class is [{pred[0]}]")
else:
    print(f"Warning: Test file not found at {test_file}")

# Verify current available classes in the dataset (Fire excluded)
csv_path = "dataSet/Metadata/Metadata V1.0 FSC22.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    available_classes = df[df["Class Name"] != "Fire"]["Class Name"].unique()
    print("\nCurrent system classes (Fire excluded):")
    print(available_classes)