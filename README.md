# Song Genre Classification

A deep learning project for classifying songs into different genres using transfer learning with Google's YAMNet model.

## 📁 Project Structure

```
songClassify/
├── data/                          # Audio dataset
│   ├── genre1/                    # Genre subdirectory
│   │   ├── song1.mp3
│   │   ├── song2.mp3
│   │   └── ...
│   ├── genre2/
│   └── ...
├── src/                           # Source code
│   ├── train.py                   # Training script
│   └── predict.py                 # Prediction/inference script
├── scripts/                       # Utility scripts
│   └── validate_data.py           # Data validation script
├── models/                        # Trained models (generated)
│   ├── best_model.h5              # Best model checkpoint
│   ├── song_classifier_yamnet.h5  # Final trained model
│   └── label_encoder_classes.npy  # Genre label mappings
├── outputs/                       # Training outputs (generated)
│   ├── X_embeddings.npy           # Cached embeddings
│   └── y_encoded.npy              # Encoded labels
├── docs/                          # Documentation
│   ├── QUICK_REFERENCE.md         # Quick start guide
│   ├── TRAINING_IMPROVEMENTS.md   # Detailed improvements
│   └── README.md                  # This file
├── dataPrep/                      # Data preparation scripts
├── .venv/                         # Python virtual environment
├── train.sh                       # Training wrapper script
├── predict.sh                     # Prediction wrapper script
├── validate.sh                    # Validation wrapper script
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your MP3 files in the following structure:

```
data/
├── genre1/
│   ├── song1.mp3
│   ├── song2.mp3
│   └── ... (95+ files recommended)
├── genre2/
│   └── ... (95+ files recommended)
└── ...
```

**Current Dataset**: 5 genres with 95+ songs each (~475+ total files)

### 3. Validate Your Data

```bash
./validate.sh
```

This will check:
- ✅ File counts per genre
- ✅ Audio file integrity
- ✅ Duration statistics
- ✅ Class balance
- ✅ Training readiness

### 4. Train the Model

```bash
./train.sh
```

Or directly:
```bash
cd src && python train.py
```

Training will:
- Load and augment audio data (3x augmentation)
- Extract YAMNet embeddings
- Train a classifier with 60/20/20 train/val/test split
- Save best model to `models/best_model.h5`
- Generate detailed evaluation metrics

### 5. Make Predictions

```bash
# Predict single file
./predict.sh path/to/song.mp3

# Predict all files in a directory
./predict.sh path/to/folder/
```

Or directly:
```bash
cd src && python predict.py path/to/song.mp3
```

## 🎯 Model Configuration

**Optimized for 475+ files with 5 genres:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Audio Duration | 10 seconds | Optimal for genre features |
| Batch Size | 32 | Good for ~475 files |
| Epochs | 100 | With early stopping |
| Augmentation | 3x per sample | 475 → ~1,425 samples |
| Learning Rate | 0.001 | Fast convergence |
| Architecture | 512→256→128 + BatchNorm | Deep with regularization |

## 📊 Expected Performance

With 475+ balanced files across 5 genres:

- **Good**: 70-80% accuracy
- **Very Good**: 80-90% accuracy
- **Excellent**: 90%+ accuracy

## 🔧 Key Features

### Data Augmentation
- Time stretching (0.9x - 1.1x)
- Pitch shifting (-2 to +2 semitones)
- Automatic class balancing with weights

### Model Architecture
- Transfer learning with Google's YAMNet
- YAMNet embeddings (1024-dim) → Classifier
- Batch Normalization for stable training
- Progressive Dropout (0.5 → 0.4 → 0.3)

### Training Features
- Proper train/val/test split (60/20/20)
- Early stopping on validation accuracy
- Learning rate reduction on plateau
- Model checkpointing (saves best model)
- Class weights for imbalanced data

### Evaluation
- Overall accuracy and loss
- Per-class accuracy breakdown
- Validation and test metrics
- Training history summary

## 📝 Generated Files

After training, you'll find:

**In `models/` directory:**
- `best_model.h5` - Best model (use for inference)
- `song_classifier_yamnet.h5` - Final model
- `label_encoder_classes.npy` - Genre label mappings

**In `outputs/` directory:**
- `X_embeddings.npy` - Cached embeddings (optional)
- `y_encoded.npy` - Encoded labels (optional)

## 💡 Tips

### For Best Results:
1. **Balanced Dataset**: Aim for equal samples per genre
2. **Quality Audio**: Consistent quality across all files
3. **Sufficient Data**: Minimum 50 files per genre, 95+ recommended
4. **Clean Labels**: Ensure files are in correct genre folders
5. **Diverse Samples**: Different artists/styles within each genre

### Troubleshooting:

**Low Accuracy?**
- Check if genres are truly distinct
- Verify no mislabeled files
- Try different DURATION (5, 10, 15 seconds)
- Reduce augmentation if too aggressive

**Out of Memory?**
- Reduce BATCH_SIZE in `src/train.py`
- Reduce DURATION
- Reduce AUGMENTATION_FACTOR

**Slow Training?**
- Check GPU availability: `nvidia-smi`
- Reduce DURATION for faster embedding extraction
- Use fewer augmentations

## 🔬 Advanced Usage

### Custom Configuration

Edit `src/train.py` to modify:

```python
# Configuration
DATA_DIR = '../data'
SAMPLE_RATE = 16000
DURATION = 10  # Adjust audio duration
BATCH_SIZE = 32  # Adjust batch size
EPOCHS = 100
AUGMENTATION_FACTOR = 3  # Adjust augmentation
```

### Using the Model in Python

```python
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model('models/best_model.h5')
labels = np.load('models/label_encoder_classes.npy')

# Make prediction (after extracting embeddings)
predictions = model.predict(embeddings)
predicted_genre = labels[np.argmax(predictions)]
```

## 📚 Documentation

- **Quick Reference**: `docs/QUICK_REFERENCE.md`
- **Training Improvements**: `docs/TRAINING_IMPROVEMENTS.md`
- **Setup Instructions**: `docs/SETUP_INSTRUCTIONS.md`

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Audio Processing**: Librosa
- **Transfer Learning**: YAMNet (TensorFlow Hub)
- **Data Science**: NumPy, Scikit-learn
- **Utilities**: tqdm, matplotlib

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Google's YAMNet model for audio embeddings
- TensorFlow Hub for pre-trained models
- Librosa for audio processing utilities

---

**Ready to train? Run `./validate.sh` first to check your data, then `./train.sh` to start training!** 🎵🚀
