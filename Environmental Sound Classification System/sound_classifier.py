"""
==============================================
ENVIRONMENTAL SOUND CLASSIFICATION SYSTEM
Using ESC-10 Dataset with Supervised Learning
==============================================

This project teaches core ML concepts:
- Sound → Numbers (Feature Extraction)
- Labeled vs Unlabeled Data
- model.fit() vs model.predict()
- Overfitting & Generalization
- Why feature quality > model complexity

Dataset: ESC-10 (10 sound classes, ~40 files each)
Classes: Dog bark, Rain, Sea waves, Baby cry, Clock tick,
         Person sneeze, Helicopter, Chainsaw, Rooster, Fire crackling

===== STEP 1: UNDERSTAND SOUND AS DATA =====
Sound is just amplitude values over time, sampled at fixed frequency (22,050 Hz).
Raw waveform is bad for ML → we extract features instead.
"""

import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# STEP 2: FEATURE EXTRACTION (MOST IMPORTANT PART)
# ============================================================
# We will NOT feed raw sound to ML model.
# Instead, extract meaningful features that ML can learn from.

def extract_features(file_path):
    """
    Extract acoustic features from an audio file.
    
    Features extracted:
    - MFCC (13): Mel-Frequency Cepstral Coefficients
              How humans perceive sound, based on ear response
    - Chroma (12): Pitch/color information, what notes are in the sound
    - Zero Crossing Rate (1): Noisiness - how fast amplitude changes sign
    - Spectral Centroid (1): Brightness of sound
    - Spectral Rolloff (1): Where most high-freq energy concentrates
    - Total: 28 features
    
    Sound (amplitude values) → 28-dimensional vector → ML model
    
    Args:
        file_path: Path to audio file (.ogg, .wav, etc.)
    
    Returns:
        numpy array of shape (28,) containing extracted features
    """
    try:
        # Load audio file
        # y = audio time series (amplitude values)
        # sr = sampling rate (samples per second)
        # sr=None means keep original sampling rate
        y, sr = librosa.load(file_path, sr=None)
        
        # Feature 1-13: MFCC (Mel-Frequency Cepstral Coefficients)
        # n_mfcc=13: extract 13 coefficients (industry standard)
        # librosa returns shape (13, time_steps) → we take mean across time
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        
        # Feature 14-25: Chroma (pitch information)
        # 12 chroma bins (musical notes)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # Feature 26: Zero Crossing Rate (noisiness)
        # Measures how frequently signal crosses zero
        # High ZCR = noisy, Low ZCR = tonal
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Feature 27: Spectral Centroid (brightness)
        # Which frequency has most energy? (like center of mass in frequency domain)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Feature 28: Spectral Rolloff (high-frequency energy concentration)
        # Below this frequency is 85% of spectral energy
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Stack all features into one vector: (28,)
        # This vector is what ML model will learn to classify
        features = np.hstack((mfcc, chroma, zcr, spectral_centroid, spectral_rolloff))
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ============================================================
# STEP 3: CREATE DATASET MATRIX
# ============================================================
# X = feature vectors (2D array: samples × features)
# y = labels (1D array: sound class for each sample)

def build_dataset(dataset_path):
    """
    Scan ESC-10 directory and extract features from all audio files.
    
    Structure:
    ESC-10/
    ├── 001 - Dog bark/
    │   ├── file1.ogg
    │   ├── file2.ogg
    │   └── ...
    ├── 002 - Rain/
    │   ├── file1.ogg
    │   └── ...
    └── ...
    
    Returns:
        X: numpy array shape (n_samples, 28) - feature vectors
        y: numpy array shape (n_samples,) - class labels (0-9)
        class_names: list of 10 sound class names
    """
    X = []  # Will store feature vectors
    y = []  # Will store labels (0-9)
    class_names = []  # Will store "Dog bark", "Rain", etc.
    
    # Find all folders in ESC-10
    dataset_path = Path(dataset_path)
    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name not in ['__pycache__']])
    
    print(f"\n{'='*60}")
    print(f"LOADING DATASET FROM: {dataset_path}")
    print(f"{'='*60}")
    
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name  # e.g., "001 - Dog bark"
        # Extract human-readable name: "Dog bark"
        readable_name = " ".join(class_name.split(" - ")[1:]) if " - " in class_name else class_name
        class_names.append(readable_name)
        
        print(f"\nClass {class_idx}: {readable_name}")
        
        # Count files in this class
        audio_files = list(class_folder.glob("*.ogg")) + list(class_folder.glob("*.wav"))
        print(f"  Found {len(audio_files)} audio files")
        
        # Extract features from each audio file
        for file_path in audio_files:
            features = extract_features(str(file_path))
            
            if features is not None:
                X.append(features)
                y.append(class_idx)  # Label: 0 for Dog bark, 1 for Rain, etc.
    
    # Convert lists to numpy arrays
    # X shape: (n_samples, 28)
    # y shape: (n_samples,)
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n{'='*60}")
    print(f"DATASET CREATED")
    print(f"{'='*60}")
    print(f"X shape: {X.shape} → ({X.shape[0]} audio files, {X.shape[1]} features)")
    print(f"y shape: {y.shape} → ({y.shape[0]} labels)")
    print(f"Classes: {class_names}")
    print(f"Data type check - X: {X.dtype}, y: {y.dtype}")
    
    return X, y, class_names


# ============================================================
# STEP 4: APPLY SUPERVISED LEARNING
# ============================================================

def train_model(X, y):
    """
    Split data into train/test and train Random Forest classifier.
    
    Key concepts:
    - train_test_split: Separate labeled data
      * 80% for training (model learns patterns from labels)
      * 20% for testing (unseen data to measure real performance)
    
    - RandomForestClassifier: Ensemble method
      * Multiple decision trees voting together
      * More stable than single tree
      * Good starting point for classification
    
    The fit() method is where LEARNING happens:
    - Model receives (X_train, y_train)
    - Adjusts internal parameters to minimize prediction error
    - After fit(), model "knows" patterns from training data
    
    Returns:
        model: Trained classifier
        X_test, y_test: Test data for evaluation
    """
    print(f"\n{'='*60}")
    print(f"TRAINING PHASE: SPLITTING DATA")
    print(f"{'='*60}")
    
    # Split: 80% train, 20% test
    # random_state=42 ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples: {len(X_test)} (20%)")
    print(f"X_train shape: {X_train.shape} → (samples, features)")
    print(f"y_train shape: {y_train.shape} → (labels)")
    
    print(f"\n{'='*60}")
    print(f"TRAINING PHASE: model.fit() - LEARNING FROM LABELED DATA")
    print(f"{'='*60}")
    print(f"Starting to learn patterns from {len(X_train)} labeled examples...")
    
    # CREATE AND TRAIN MODEL
    # This is where the learning happens!
    model = RandomForestClassifier(
        n_estimators=100,  # 100 decision trees voting together
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    # FIT: Learn patterns from labeled training data
    # X_train = feature vectors
    # y_train = correct labels
    # Model adjusts itself to map X_train → y_train
    model.fit(X_train, y_train)
    
    print(f"✓ Training complete!")
    print(f"Model has learned decision boundaries from labeled training data")
    
    # Training accuracy (how well on seen data)
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\nTraining accuracy: {train_accuracy:.2%}")
    print(f"(How well model performs on data it learned from)")
    
    return model, X_test, y_test, X_train, y_train


# ============================================================
# STEP 5: PREDICTION (REALITY CHECK)
# ============================================================

def make_predictions(model, X_test, y_test, class_names):
    """
    Use trained model to predict on unseen test data.
    
    This is where the magic happens:
    - X_test has NO labels (y_test is hidden from model)
    - Model applies learned patterns
    - Generates predictions completely independently
    - We then check if predictions match y_test
    
    predict() is the opposite of fit():
    - fit(): Learn (X, y) → (X, y)
    - predict(): Apply learning (X only) → ŷ (predicted labels)
    
    Returns:
        predictions: Array of predicted labels
    """
    print(f"\n{'='*60}")
    print(f"PREDICTION PHASE: model.predict() - CLASSIFY UNSEEN DATA")
    print(f"{'='*60}")
    print(f"Now using model on {len(X_test)} NEW audio files...")
    print(f"These files were NOT used during training!")
    print(f"Model doesn't know correct labels (y_test) - it must guess!")
    
    # PREDICT: Apply learned patterns to new data
    # Input: X_test (features only, no labels)
    # Output: predicted labels (0-9)
    # Process: Model uses learned decision boundaries
    predictions = model.predict(X_test)
    
    print(f"\n✓ Predictions complete!")
    print(f"Predicted labels shape: {predictions.shape}")
    print(f"Sample predictions (first 10):")
    print(f"  Actual:    {y_test[:10]}")
    print(f"  Predicted: {predictions[:10]}")
    
    return predictions


# ============================================================
# STEP 6: EVALUATE MODEL PERFORMANCE
# ============================================================

def evaluate_model(model, X_test, y_test, predictions, class_names):
    """
    Measure how well predictions match actual labels.
    
    Key insight:
    - We can only evaluate on test set (where we know y_test)
    - This tells us real performance on unseen data
    - Not testing on training data (that's cheating!)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION PHASE: DID MODEL LEARN CORRECTLY?")
    print(f"{'='*60}")
    
    # Overall accuracy
    test_accuracy = accuracy_score(y_test, predictions)
    print(f"\n🎯 TEST ACCURACY: {test_accuracy:.2%}")
    print(f"   Out of {len(y_test)} unseen test samples, {np.sum(predictions == y_test)} were correct")
    print(f"   This is REAL performance on completely new data!")
    
    # Detailed report per class
    print(f"\n{'='*60}")
    print(f"DETAILED PERFORMANCE BY SOUND CLASS")
    print(f"{'='*60}")
    print(classification_report(y_test, predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX (Where model gets confused)")
    print(f"{'='*60}")
    print(f"Rows = Actual class, Columns = Predicted class")
    print(f"Diagonal = Correct predictions, Off-diagonal = Mistakes\n")
    print(cm)
    
    # Which classes are confused?
    print(f"\n{'='*60}")
    print(f"INSIGHT: WHERE DOES MODEL STRUGGLE?")
    print(f"{'='*60}")
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for idx, (class_name, accuracy) in enumerate(zip(class_names, per_class_accuracy)):
        print(f"{class_name:25} → {accuracy:.2%} correct")
    
    return test_accuracy, cm


def plot_results(cm, class_names, accuracy):
    """
    Visualize confusion matrix and performance.
    """
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATION")
    print(f"{'='*60}")
    
    plt.figure(figsize=(12, 10))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Sound Classification - Confusion Matrix\nTest Accuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENVIRONMENTAL SOUND CLASSIFICATION SYSTEM")
    print("Using ESC-10 Dataset")
    print("="*60)
    
    # Path to ESC-10 dataset
    dataset_path = Path(__file__).parent / "ESC-10" / "ESC-10"
    
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print(f"Please ensure ESC-10 folder exists in the project directory")
        exit(1)
    
    # STEP 3: Load and extract features
    print("\n" + "="*60)
    print("STEP 1-3: LOADING DATASET & EXTRACTING FEATURES")
    print("="*60)
    X, y, class_names = build_dataset(dataset_path)
    
    # STEP 4: Train model
    print("\n" + "="*60)
    print("STEP 4: TRAINING MODEL")
    print("="*60)
    model, X_test, y_test, X_train, y_train = train_model(X, y)
    
    # STEP 5: Make predictions
    print("\n" + "="*60)
    print("STEP 5: MAKING PREDICTIONS")
    print("="*60)
    predictions = make_predictions(model, X_test, y_test, class_names)
    
    # STEP 6: Evaluate
    print("\n" + "="*60)
    print("STEP 6: EVALUATING PERFORMANCE")
    print("="*60)
    test_accuracy, cm = evaluate_model(model, X_test, y_test, predictions, class_names)
    
    # Visualize
    plot_results(cm, class_names, test_accuracy)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROJECT SUMMARY")
    print(f"{'='*60}")
    print(f"""
✓ Sound → Numbers: Extracted {X.shape[1]} features per audio file
✓ Labeled Data: Used {len(X)} labeled sound samples
✓ Supervised Learning: model.fit() learned from labeled data
✓ Unseen Data: model.predict() on {len(X_test)} new samples
✓ Generalization: {test_accuracy:.2%} accuracy on unseen data
✓ Feature Quality: 28 acoustic features > raw waveform

KEY INSIGHTS:
1. Features matter more than algorithm
2. fit() = Learning from labels
3. predict() = Apply learning without labels
4. Train/Test split prevents overfitting

NEXT STEPS:
- Convert to unsupervised learning (KMeans clustering)
- Try different classifiers (SVM, Neural Networks)
- Experiment with different features
- Analyze which features are most important
    """)
