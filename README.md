# Music Genre Classification Project

## Overview

This project focuses on classifying music genres using audio signal processing and machine learning techniques. The dataset used for this task is the **GTZAN Dataset**, which is widely regarded as the “MNIST for music.” It consists of 1000 audio tracks across 10 different genres, with each track lasting for 30 seconds. The goal is to classify these tracks into their respective genres.

The project follows a step-by-step approach, including signal processing, feature extraction, and machine learning model development.

---

## Dataset

The **GTZAN Dataset** contains:

- 1000 music tracks
- 10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
- Each track is 30 seconds long

To enhance the dataset size, each track is divided into **10 segments** of 3 seconds each, increasing the data size to 10,000 segments.

---

## Signal Processing

### Features Extracted

The following audio features are extracted from each 3-second segment:

1. **Chromagram**: Represents the energy distribution of the 12 distinct pitch classes.
2. **RMS Energy**: Measures perceived loudness.
3. **Spectral Centroid**: Indicates how bright or dark a sound is.
4. **Spectral Bandwidth**: Shows the width of the frequency band.
5. **Spectral Rolloff**: The frequency below which a specified percentage (e.g., 85%) of the total spectral energy is contained.
6. **Zero-Crossing Rate (ZCR)**: The rate at which the signal changes sign.
7. **Harmonic and Percussive Source Separation (HPSS)**: Separates harmonic and percussive elements.
8. **Mel-Frequency Cepstral Coefficients (MFCC)**: Extracts the short-term power spectrum of a sound, useful for speech and audio classification.

---

## Machine Learning Models

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

knn_cls = KNeighborsClassifier(n_neighbors=3)
knn_cls.fit(X_train, y_train)
y_pred = knn_cls.predict(X_test)
sys.exit()

### How to run
```python
pip install librosa torch sklearn #Install required libraries
!gdown 1MGhyeMngD6P9Kz9zJpL68ylQaIQvW7Zx #Download and unzip the GTZAN dataset
!unzip GTZAN.zip -d /content/GTZAN
!rm GTZAN.zip
# Example of extracting chromagram # Preprocess and extract features
chromagram = librosa.feature.chroma_stft(y=audio_segment, sr=sample_rate)
# Train KNN model
knn_cls.fit(X_train, y_train)
# Evaluate model
accuracy = knn_cls.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
sys.exit()

## Conclusion
This README includes sections for an overview, dataset, features, machine learning models, and instructions on how to run the project. You can adjust or extend it based on additional details or requirements specific to your project!
