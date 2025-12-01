#!/usr/bin/env python3
"""
Inference Script for Song Genre Classification

Usage:
    python predict.py path/to/song.mp3
    python predict.py path/to/folder/  # Predict all MP3s in folder
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Configuration (should match training)
MODELS_DIR = '../models'
SAMPLE_RATE = 16000
DURATION = 10
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

def load_and_preprocess_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio file."""
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        target_length = sr * duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_yamnet_embedding(yamnet_model, audio):
    """Extract YAMNet embedding from audio."""
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    avg_embedding = tf.reduce_mean(embeddings, axis=0)
    return avg_embedding.numpy()

def predict_genre(model, yamnet_model, labels, audio_path):
    """Predict genre for a single audio file."""
    # Load and preprocess
    audio = load_and_preprocess_audio(audio_path)
    if audio is None:
        return None
    
    # Extract embedding
    embedding = extract_yamnet_embedding(yamnet_model, audio)
    embedding = np.expand_dims(embedding, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(embedding, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    return {
        'genre': labels[predicted_idx],
        'confidence': confidence,
        'all_probabilities': {labels[i]: predictions[0][i] for i in range(len(labels))}
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file.mp3 or directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Model paths
    model_path = os.path.join(MODELS_DIR, 'best_model.h5')
    label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder_classes.npy')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please train the model first using: python train.py")
        sys.exit(1)
    
    if not os.path.exists(label_encoder_path):
        print(f"❌ Error: Label encoder file '{label_encoder_path}' not found!")
        sys.exit(1)
    
    print("Loading model and YAMNet...")
    model = load_model(model_path)
    labels = np.load(label_encoder_path)
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    print("✅ Models loaded successfully!\n")
    
    # Get list of files to process
    if os.path.isfile(input_path):
        if input_path.endswith('.mp3'):
            files = [input_path]
        else:
            print("❌ Error: File must be an MP3 file")
            sys.exit(1)
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.mp3')]
        if not files:
            print(f"❌ Error: No MP3 files found in {input_path}")
            sys.exit(1)
    else:
        print(f"❌ Error: {input_path} not found")
        sys.exit(1)
    
    print(f"Processing {len(files)} file(s)...\n")
    print("="*70)
    
    # Process each file
    results = []
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\n🎵 {filename}")
        
        result = predict_genre(model, yamnet_model, labels, file_path)
        
        if result:
            print(f"   Predicted Genre: {result['genre']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print(f"   All Probabilities:")
            for genre, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"      {genre}: {prob*100:.2f}%")
            results.append((filename, result))
        else:
            print(f"   ❌ Failed to process")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(results)}/{len(files)}")
    
    if results:
        genre_counts = {}
        for _, result in results:
            genre = result['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        print("\nPredicted Genre Distribution:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {genre}: {count} file(s)")
        
        avg_confidence = np.mean([r['confidence'] for _, r in results])
        print(f"\nAverage Confidence: {avg_confidence*100:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
