# “Environmental Sound Classification System”

Dataset link : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT
(Classify sounds like siren, dog bark, drilling, engine, rain, crowd noise, etc.)

This is a real ML project, used in:

smart cities

surveillance

accessibility apps

IoT devices

What You Will Learn (Core ML Concepts)

Sound → numbers (feature extraction)

Labeled vs unlabeled data

model.fit() vs model.predict()

Overfitting & generalization

Why feature quality > model complexity

Dataset (Don’t create chaos, start clean)

Use ESC-50 dataset (industry standard)

2,000 audio clips

5 sec each

50 classes

WAV format

Labeled → supervised learning

(You can later convert same project to unsupervised)

Step-by-Step Execution Plan
STEP 1: Understand sound as data

Sound is just:

amplitude values over time

sampled at fixed frequency (e.g., 22,050 Hz)

Raw waveform is bad for ML, so we extract features.

STEP 2: Feature Extraction (Most Important Part)

We will not feed raw sound to ML model.

We extract:

Feature	Meaning
MFCC	How humans perceive sound
Chroma	Pitch information
Spectral Centroid	Brightness of sound
Zero Crossing Rate	Noisiness
Spectral Roll-off	High-frequency energy

Use librosa.

Example:

import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    return np.hstack((mfcc, chroma, zcr))


Now sound → numeric vector

This is where ML actually starts.

STEP 3: Create Dataset Matrix

X → feature vectors

y → labels (sound type)

X shape = (samples, features)
y shape = (samples,)

STEP 4: Apply Supervised Learning

Start simple.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier()
model.fit(X_train, y_train)


This is pure learning.

STEP 5: Prediction (Reality Check)
pred = model.predict(X_test)


Now:

No labels used

Model just applies learned patterns

Exactly what you were asking about earlier.

STEP 6: Evaluate
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))


Now you’ll understand:

Why some sounds are misclassified

Why feature quality matters

Where Your Concepts Fit Perfectly
fit vs predict
Phase	Meaning
fit	Learn patterns from labeled sound
predict	Classify unseen sound
Supervised vs Unsupervised

ESC-50 → supervised

Remove labels + use KMeans → unsupervised sound clustering

Extension (Very Important for Learning)

After supervised version, do this:

Convert to Unsupervised
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
labels = kmeans.predict(X)


Now you’ll feel the difference instead of memorizing it.

Final Learning Outcome (You’ll genuinely understand ML)

After this project, you’ll understand:

Why raw data ≠ ML ready data

Why fit() is learning

Why predict() works without labels

Why features matter more than algorithms

Optional Advanced Version (If you want depth)

CNN on spectrogram images

Real-time microphone input

Noise reduction

Deploy as web app

Final Recommendation (Very Important)

Start with:

Classical ML + features
Not deep learning.

Deep learning hides understanding.
