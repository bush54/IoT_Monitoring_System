import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# File paths configuration
csv_path = "dataSet/Metadata/Metadata V1.0 FSC22.csv"
audio_folder = "dataSet/AudioWise"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(f"Metadata file not found at: {csv_path}")

# Rename columns for consistency
df = df.rename(columns={"Dataset File Name": "filename", "Class Name": "target"})

# --- Crucial Step: Remove 'Fire' class permanently ---
df = df[df["target"] != "Fire"].reset_index(drop=True)
print(f"Fire class removed. Remaining classes: {df['target'].unique()}")

df["path"] = df["filename"].apply(lambda x: os.path.join(audio_folder, x))
print("File existence check:")
print(df["path"].apply(os.path.exists).value_counts())

def extract_features_from_signal(signal, sr=16000, duration=5):
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

    return np.concatenate([
        mfcc_mean, mfcc_std, 
        [np.mean(zcr), np.std(zcr), np.mean(centroid), np.std(centroid),
         np.mean(bandwidth), np.std(bandwidth), np.mean(rolloff), np.std(rolloff),
         np.mean(rms), np.std(rms)]
    ])

def add_noise(signal, noise_factor=0.003):
    return signal + noise_factor * np.random.randn(len(signal))

def shift_signal(signal, shift_max=0.1):
    shift = np.random.randint(int(len(signal) * shift_max))
    return np.roll(signal, shift)

X, y = [], []
print("Starting feature extraction process (this may take a while)...")
for i, row in df.iterrows():
    file_path = row["path"]
    target = row["target"]
    try:
        signal, sr = librosa.load(file_path, sr=16000, mono=True)
        # Augmentation: Original, Noisy, Shifted
        X.append(extract_features_from_signal(signal, sr=sr))
        y.append(target)
        X.append(extract_features_from_signal(add_noise(signal), sr=sr))
        y.append(target)
        X.append(extract_features_from_signal(shift_signal(signal), sr=sr))
        y.append(target)
    except Exception:
        continue
    if (i + 1) % 200 == 0: 
        print(f"Processed {i+1} files...")

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Starting Model Training (GridSearchCV)...")
param_grid = {'C': [1, 10, 50], 'gamma': ['scale', 0.01], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

model = grid.best_estimator_
print("Training finished successfully without Fire class.")

# Save Model and Scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/audio_svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and Scaler saved in 'models/' directory.")

# Evaluation
y_pred = model.predict(X_test)
print(f"Final Model Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))