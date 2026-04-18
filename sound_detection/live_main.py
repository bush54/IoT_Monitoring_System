import sounddevice as sd
import numpy as np
import joblib
import librosa
import time

# 1. Load Models and Scaler
try:
    model = joblib.load('models/audio_svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ System Ready: Models and Scaler Synchronized.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Parameters (Must strictly match your Training script settings)
FS = 16000
THRESHOLD = 0.0001 
DURATION = 5

def extract_features_like_training(signal, sr=16000, duration=5):
    """
    Mirroring the exact feature extraction logic from train.py
    Order: 13 MFCC Means -> 13 MFCC Stds -> 10 Statistical Features
    Total: 36 features
    """
    # Fix duration/length to match training
    target_length = sr * duration
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))
    else:
        signal = signal[:target_length]

    # Extract core features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    
    zcr = librosa.feature.zero_crossing_rate(signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    rms = librosa.feature.rms(y=signal)

    # Combine in the exact sequence used during training
    features = np.concatenate([
        mfcc_mean,                  # 13 features
        mfcc_std,                   # 13 features
        [np.mean(zcr), np.std(zcr), 
         np.mean(centroid), np.std(centroid),
         np.mean(bandwidth), np.std(bandwidth), 
         np.mean(rolloff), np.std(rolloff),
         np.mean(rms), np.std(rms)] # 10 features
    ])
    
    return features.reshape(1, -1)

print("\n" + "="*50)
print("      AI SOUND DETECTION SYSTEM - LIVE MONITORING      ")
print("="*50)
print(f"Status: Listening for events (Threshold: {THRESHOLD})...")

try:
    while True:
        # Step 1: Background Monitoring
        check_clip = sd.rec(int(0.5 * FS), samplerate=FS, channels=1)
        sd.wait()
        volume = np.linalg.norm(check_clip) / np.sqrt(len(check_clip))
        
        # Real-time Volume Feedback
        print(f"Level: {volume:.5f} | Threshold: {THRESHOLD}", end='\r')
        
        # Step 2: Event Trigger
        if volume > THRESHOLD:
            print(f"\n\n[EVENT] Sound Detected ({volume:.4f})! Processing...")
            
            # Record fixed duration for analysis
            recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
            sd.wait()
            
            # Step 3: Analysis and Prediction
            signal = recording.flatten()
            features = extract_features_like_training(signal, FS, DURATION)
            
            # Scaling using the loaded scaler
            features_scaled = scaler.transform(features)
            
            # Final Classification
            prediction = model.predict(features_scaled)[0]
            
            print(f"🎯 ANALYSIS RESULT: [ {prediction} ]")
            print("-" * 40)
            print("[WAITING] Back to monitor mode...")
            
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n[EXIT] Monitoring stopped by user.")
except Exception as e:
    print(f"\n[CRITICAL ERROR] {e}")