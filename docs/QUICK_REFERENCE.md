# Quick Reference Guide

## 🚀 Quick Start (Once You Have 500+ Files)

### 1. Validate Your Data
```bash
source .venv/bin/activate
python validate_data.py
```

This will check:
- ✅ File counts per genre
- ✅ Audio file integrity
- ✅ Duration statistics
- ✅ Class balance
- ✅ Training readiness

### 2. Run Training
```bash
source .venv/bin/activate
python train.py
```

### 3. Monitor Progress
The training will show:
- Data loading progress
- Augmentation stats
- Class distribution
- Training/validation metrics per epoch
- Final evaluation results

---

## 📁 Expected Data Structure

```
songClassify/
├── data/
│   ├── bhojpuri/
│   │   ├── song1.mp3
│   │   ├── song2.mp3
│   │   └── ... (150+ files recommended)
│   ├── newari/
│   │   ├── song1.mp3
│   │   ├── song2.mp3
│   │   └── ... (150+ files recommended)
│   └── tamang selo/
│       ├── song1.mp3
│       ├── song2.mp3
│       └── ... (150+ files recommended)
├── train.py
├── validate_data.py
└── ...
```

---

## 🎯 Key Improvements Made

| Issue | Before | After |
|-------|--------|-------|
| Learning Rate | 0.00001 | 0.001 (100x faster) |
| Epochs | 300 | 100 (with early stopping) |
| Audio Duration | 3s | 10s |
| Data Split | Train/Test only | Train/Val/Test (60/20/20) |
| Augmentation | None | 5x per sample |
| Architecture | Simple | BatchNorm + Deeper |
| Class Weights | No | Yes (balanced) |
| Batch Size | 16 | 8 (better for small data) |

---

## 📊 What to Expect

### With Current Data (9 files, 1 class):
- ❌ Cannot train multi-class classifier
- Only one genre has data

### With 500+ Balanced Files:
- ✅ Expected accuracy: 70-90%
- ✅ Proper multi-class classification
- ✅ Reliable model performance

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError"
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Problem: "No valid files found"
- Check that MP3 files are in genre subdirectories
- Run `validate_data.py` to identify issues

### Problem: "Out of memory"
- Reduce `BATCH_SIZE` in train.py
- Reduce `DURATION` in train.py
- Reduce `AUGMENTATION_FACTOR` in train.py

### Problem: Low accuracy even with good data
- Check if genres are truly distinct
- Try different `DURATION` values (10, 15, 20s)
- Reduce augmentation if too aggressive
- Check for mislabeled files

---

## 📝 Files Generated After Training

| File | Purpose |
|------|---------|
| `best_model.h5` | Best model (use this for inference) |
| `song_classifier_yamnet.h5` | Final model |
| `label_encoder_classes.npy` | Genre label mappings |
| `X_embeddings.npy` | Cached embeddings (optional) |
| `y_encoded.npy` | Encoded labels (optional) |

---

## 🎵 Using the Trained Model

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model('best_model.h5')
labels = np.load('label_encoder_classes.npy')

# Load and preprocess audio (you'll need to extract embeddings)
# ... (similar to training pipeline)

# Predict
predictions = model.predict(embeddings)
predicted_genre = labels[np.argmax(predictions)]
print(f"Predicted genre: {predicted_genre}")
```

---

## 💡 Tips for Data Collection

1. **Quality over Quantity**: 
   - Better to have 100 high-quality, representative songs per genre
   - Than 500 low-quality or mislabeled songs

2. **Consistency**:
   - Similar audio quality across genres
   - Full songs or consistent clip lengths
   - Avoid live recordings mixed with studio recordings

3. **Balance**:
   - Aim for equal samples per genre
   - Minimum 50 files per genre
   - Recommended 150+ files per genre

4. **Diversity**:
   - Different artists per genre
   - Different eras/styles within genre
   - Avoid duplicates or very similar songs

---

## 🚦 Current Status

- ✅ Training code fixed and optimized
- ✅ Data validation script created
- ✅ Documentation complete
- ⏳ Waiting for 500+ MP3 files
- ⏳ Ready to train once data is available

---

**Good luck with data collection! Run `validate_data.py` once you have the files to check everything is ready.** 🎉
