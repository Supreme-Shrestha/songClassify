import os
import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm

# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuration
DATA_DIR = '../data'
OUTPUT_DIR = '../processed_data'
SAMPLE_RATE = 22050
DURATION = 3 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
BATCH_SIZE = 32 # Process audio segments in batches on GPU

@tf.function
def compute_spectrogram_tf(signals):
    """
    Computes Log-Mel Spectrogram using TensorFlow (runs on GPU).
    Args:
        signals: Tensor of shape [batch_size, samples]
    Returns:
        log_mel_spectrograms: Tensor of shape [batch_size, time_steps, n_mels]
    """
    # STFT parameters matching librosa defaults roughly
    frame_length = 2048
    frame_step = 512
    fft_length = 2048
    n_mels = 128
    
    # 1. STFT
    stft = tf.signal.stft(signals, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrograms = tf.abs(stft)
    
    # 2. Mel Matrix
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2.0
    )
    
    # 3. Mel Spectrogram
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1] + (n_mels,))
    
    # 4. Log (Power to DB-like)
    # Add small epsilon to avoid log(0)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    
    return log_mel_spectrograms

def save_batch(spectrograms, base_names, segment_indices, output_dir):
    """
    Saves a batch of spectrograms to disk.
    """
    spectrograms_np = spectrograms.numpy()
    
    # Normalize per batch or globally? 
    # Usually better to normalize per sample or use a fixed global scaler.
    # Here we do min-max per sample to match previous logic
    for i in range(len(spectrograms_np)):
        spec = spectrograms_np[i]
        
        # Normalize 0-1
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 0:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = np.zeros_like(spec)
            
        save_name = f"{base_names[i]}_segment_{segment_indices[i]}.npy"
        save_path = os.path.join(output_dir, save_name)
        np.save(save_path, spec)

def preprocess_dataset(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found genres: {genres}")

    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        output_genre_dir = os.path.join(output_dir, genre)
        if not os.path.exists(output_genre_dir):
            os.makedirs(output_genre_dir)
            
        print(f"Processing {genre}...")
        files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3')]
        
        # Buffers for batching
        batch_signals = []
        batch_names = []
        batch_indices = []
        
        for file_name in tqdm(files):
            file_path = os.path.join(genre_dir, file_name)
            try:
                # Load audio (CPU - bottleneck usually here)
                # librosa.load is robust for MP3s
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Split into segments
                num_segments = int(len(signal) / SAMPLES_PER_TRACK)
                
                for s in range(num_segments):
                    start_sample = SAMPLES_PER_TRACK * s
                    finish_sample = start_sample + SAMPLES_PER_TRACK
                    
                    segment = signal[start_sample:finish_sample]
                    
                    # Add to batch
                    batch_signals.append(segment)
                    batch_names.append(os.path.splitext(file_name)[0])
                    batch_indices.append(s)
                    
                    # Process batch if full
                    if len(batch_signals) >= BATCH_SIZE:
                        # Convert to tensor
                        signals_tensor = tf.convert_to_tensor(np.array(batch_signals), dtype=tf.float32)
                        
                        # Compute on GPU
                        specs = compute_spectrogram_tf(signals_tensor)
                        
                        # Save
                        save_batch(specs, batch_names, batch_indices, output_genre_dir)
                        
                        # Clear buffers
                        batch_signals = []
                        batch_names = []
                        batch_indices = []
                        
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        
        # Process remaining items in the last batch
        if batch_signals:
            signals_tensor = tf.convert_to_tensor(np.array(batch_signals), dtype=tf.float32)
            specs = compute_spectrogram_tf(signals_tensor)
            save_batch(specs, batch_names, batch_indices, output_genre_dir)

if __name__ == "__main__":
    # Ensure paths are correct relative to script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data')
    output_path = os.path.join(os.path.dirname(current_dir), 'processed_data')
    
    preprocess_dataset(data_path, output_path)
