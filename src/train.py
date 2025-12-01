import os
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import librosa
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuration
DATA_DIR = '../data'
MODELS_DIR = '../models'
OUTPUTS_DIR = '../outputs'
SAMPLE_RATE = 16000  # YAMNet requires 16kHz
DURATION = 10  # seconds - optimal for genre classification
BATCH_SIZE = 32  # Optimal for dataset with 475+ files
EPOCHS = 100  # With early stopping
AUGMENTATION_FACTOR = 3  # Create 3 augmented versions per sample (475 -> ~1900 samples)

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# YAMNet model URL
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

def augment_audio(y, sr):
    """Apply random augmentations to audio."""
    augmented = []
    
    # Original
    augmented.append(y)
    
    # Time stretching
    y_stretch = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
    if len(y_stretch) > len(y):
        y_stretch = y_stretch[:len(y)]
    else:
        y_stretch = np.pad(y_stretch, (0, len(y) - len(y_stretch)))
    augmented.append(y_stretch)
    
    # Pitch shifting
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-2, 3))
    augmented.append(y_pitch)
    
    return augmented

def load_and_preprocess_audio(file_path, sr=SAMPLE_RATE, duration=DURATION, augment=False):
    """Load audio file and prepare for YAMNet."""
    try:
        # Load audio
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad if too short
        target_length = sr * duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        if augment:
            return augment_audio(y, sr)
        return [y]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_yamnet_embeddings(yamnet_model, audio_data):
    """
    Extract embeddings from YAMNet for a batch of audio samples.
    YAMNet outputs embeddings for each 0.96s window, we'll average them.
    """
    embeddings_list = []
    
    print("Extracting embeddings...")
    for audio in tqdm(audio_data, desc="Processing audio files"):
        # YAMNet expects float32 waveform
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Get embeddings (scores, embeddings, spectrogram)
        _, embeddings, _ = yamnet_model(waveform)
        
        # Average embeddings across time
        avg_embedding = tf.reduce_mean(embeddings, axis=0)
        embeddings_list.append(avg_embedding.numpy())
    
    return np.array(embeddings_list)

def load_data(data_dir, augment=False):
    """Load all audio files and their labels."""
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found!\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please ensure the data directory exists with genre subdirectories."
        )
    
    X = []
    y = []
    
    genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not genres:
        raise ValueError(
            f"No subdirectories found in '{data_dir}'!\n"
            f"Expected structure: {data_dir}/genre_name/*.mp3\n"
            f"Contents of {data_dir}: {os.listdir(data_dir)}"
        )
    
    print(f"Classes: {genres}")
    
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3')]
        
        print(f"Loading {len(files)} files from {genre}...")
        for i, f in enumerate(files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(files)}")
            
            file_path = os.path.join(genre_dir, f)
            audio_list = load_and_preprocess_audio(file_path, augment=augment)
            
            if audio_list is not None:
                for audio in audio_list:
                    X.append(audio)
                    y.append(genre)
    
    print(f"\nTotal samples loaded: {len(X)}")
    return np.array(X), np.array(y)

def create_transfer_learning_model(yamnet_model, num_classes):
    """
    Create a transfer learning model using YAMNet embeddings.
    We'll freeze YAMNet and add a classification head.
    """
    # Input layer for audio waveform
    input_audio = Input(shape=(SAMPLE_RATE * DURATION,), dtype=tf.float32, name='audio_input')
    
    # Get YAMNet embeddings (freeze the base model)
    _, embeddings, _ = yamnet_model(input_audio)
    
    # Average embeddings across time windows
    avg_embedding = tf.reduce_mean(embeddings, axis=0)
    
    # Classification head
    x = Dense(256, activation='relu')(avg_embedding)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_audio, outputs=predictions)
    
    # Freeze YAMNet layers
    yamnet_model.trainable = False
    
    return model

def main():
    print("Loading YAMNet model from TensorFlow Hub...")
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    print("YAMNet loaded successfully!")
    
    print("\n" + "="*60)
    print("LOADING DATA WITH AUGMENTATION")
    print("="*60)
    print("\nLoading training data with augmentation...")
    X_audio, y = load_data(DATA_DIR, augment=True)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"\nData shape: {X_audio.shape}")
    print(f"Labels shape: {y_encoded.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")
    
    # Check class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(le.classes_, counts):
        print(f"  {cls}: {count} samples")
    
    # Extract YAMNet embeddings for all audio
    print("\n" + "="*60)
    print("EXTRACTING YAMNET EMBEDDINGS")
    print("="*60)
    X_embeddings = extract_yamnet_embeddings(yamnet_model, X_audio)
    print(f"Embeddings shape: {X_embeddings.shape}")
    
    # Split data: 60% train, 20% val, 20% test
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_embeddings, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 of 0.8 = 0.2
    )
    
    print(f"Train set: {X_train.shape} ({len(X_train)/len(X_embeddings)*100:.1f}%)")
    print(f"Validation set: {X_val.shape} ({len(X_val)/len(X_embeddings)*100:.1f}%)")
    print(f"Test set: {X_test.shape} ({len(X_test)/len(X_embeddings)*100:.1f}%)")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nClass weights (for handling imbalanced data):")
    for i, weight in class_weight_dict.items():
        print(f"  {le.classes_[i]}: {weight:.2f}")
    
    # Create improved classifier on top of embeddings
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    input_embedding = Input(shape=(X_embeddings.shape[1],), name='embedding_input')
    x = Dense(512, activation='relu')(input_embedding)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_embedding, outputs=predictions)
    
    # Increased learning rate for better convergence
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Increased from 0.00001
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Changed to val_accuracy
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,  # Handle class imbalance
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Detailed predictions on test set
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nPer-class accuracy on test set:")
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred_classes[mask] == y_test[mask]).mean()
            print(f"  {cls}: {acc:.4f} ({mask.sum()} samples)")
    
    # Save final model
    model_path = os.path.join(MODELS_DIR, 'song_classifier_yamnet.h5')
    model.save(model_path)
    print("\n" + "="*60)
    print(f"Model saved to {model_path}")
    
    # Save label encoder
    label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder_classes.npy')
    np.save(label_encoder_path, le.classes_)
    print(f"Label encoder saved to {label_encoder_path}")
    
    # Save embeddings for future use (optional)
    embeddings_path = os.path.join(OUTPUTS_DIR, 'X_embeddings.npy')
    labels_path = os.path.join(OUTPUTS_DIR, 'y_encoded.npy')
    np.save(embeddings_path, X_embeddings)
    np.save(labels_path, y_encoded)
    print(f"Embeddings saved to {OUTPUTS_DIR}")
    print("="*60)
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best training accuracy: {max(history.history['accuracy']):.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

