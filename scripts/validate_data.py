#!/usr/bin/env python3
"""
Data Validation Script for Song Classification Dataset

This script checks:
1. Number of files per genre
2. Audio file integrity
3. Audio duration statistics
4. Sample rate consistency
5. Recommendations for training
"""

import os
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'
MIN_FILES_PER_CLASS = 50
RECOMMENDED_FILES_PER_CLASS = 150

def check_audio_file(file_path):
    """Check if audio file is valid and return its properties."""
    try:
        y, sr = librosa.load(file_path, sr=None, duration=1)  # Load only 1 second for quick check
        duration = librosa.get_duration(path=file_path)
        return {
            'valid': True,
            'duration': duration,
            'sample_rate': sr,
            'error': None
        }
    except Exception as e:
        return {
            'valid': False,
            'duration': 0,
            'sample_rate': 0,
            'error': str(e)
        }

def main():
    print("="*70)
    print("SONG CLASSIFICATION DATASET VALIDATION")
    print("="*70)
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory '{DATA_DIR}' not found!")
        return
    
    # Get all genre directories
    genres = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not genres:
        print(f"❌ Error: No genre subdirectories found in '{DATA_DIR}'!")
        return
    
    print(f"\n📁 Found {len(genres)} genre(s): {', '.join(genres)}\n")
    
    # Statistics
    stats = defaultdict(lambda: {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'durations': [],
        'sample_rates': [],
        'errors': []
    })
    
    total_files = 0
    
    # Check each genre
    for genre in sorted(genres):
        genre_dir = os.path.join(DATA_DIR, genre)
        files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3')]
        
        print(f"🎵 Checking {genre}...")
        print(f"   Found {len(files)} MP3 files")
        
        stats[genre]['total'] = len(files)
        total_files += len(files)
        
        # Check each file
        for i, filename in enumerate(files):
            file_path = os.path.join(genre_dir, filename)
            result = check_audio_file(file_path)
            
            if result['valid']:
                stats[genre]['valid'] += 1
                stats[genre]['durations'].append(result['duration'])
                stats[genre]['sample_rates'].append(result['sample_rate'])
            else:
                stats[genre]['invalid'] += 1
                stats[genre]['errors'].append((filename, result['error']))
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Validated {i + 1}/{len(files)} files...")
        
        print(f"   ✅ Valid: {stats[genre]['valid']}, ❌ Invalid: {stats[genre]['invalid']}")
        print()
    
    # Summary Report
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total files: {total_files}")
    print(f"   Total genres: {len(genres)}")
    print(f"   Average files per genre: {total_files/len(genres):.1f}")
    
    # Per-genre statistics
    print(f"\n📈 Per-Genre Statistics:")
    print(f"{'Genre':<20} {'Total':<10} {'Valid':<10} {'Invalid':<10} {'Avg Duration':<15}")
    print("-" * 70)
    
    for genre in sorted(genres):
        s = stats[genre]
        avg_duration = np.mean(s['durations']) if s['durations'] else 0
        print(f"{genre:<20} {s['total']:<10} {s['valid']:<10} {s['invalid']:<10} {avg_duration:<15.1f}s")
    
    # Check for class imbalance
    print(f"\n⚖️  Class Balance Analysis:")
    file_counts = [stats[g]['valid'] for g in genres]
    if file_counts:
        min_files = min(file_counts)
        max_files = max(file_counts)
        imbalance_ratio = max_files / min_files if min_files > 0 else float('inf')
        
        print(f"   Min files: {min_files}")
        print(f"   Max files: {max_files}")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 3:
            print(f"   ⚠️  WARNING: High class imbalance detected!")
            print(f"   Consider balancing your dataset or using class weights.")
        elif imbalance_ratio > 1.5:
            print(f"   ⚠️  Moderate class imbalance detected.")
        else:
            print(f"   ✅ Dataset is well balanced!")
    
    # Check minimum requirements
    print(f"\n✅ Dataset Readiness:")
    ready_for_training = True
    
    for genre in genres:
        valid_count = stats[genre]['valid']
        if valid_count == 0:
            print(f"   ❌ {genre}: No valid files! Cannot train.")
            ready_for_training = False
        elif valid_count < MIN_FILES_PER_CLASS:
            print(f"   ⚠️  {genre}: Only {valid_count} files (minimum {MIN_FILES_PER_CLASS} recommended)")
            ready_for_training = False
        elif valid_count < RECOMMENDED_FILES_PER_CLASS:
            print(f"   ⚠️  {genre}: {valid_count} files (recommended {RECOMMENDED_FILES_PER_CLASS}+)")
        else:
            print(f"   ✅ {genre}: {valid_count} files - Good!")
    
    # Audio quality checks
    print(f"\n🔊 Audio Quality:")
    all_sample_rates = []
    all_durations = []
    
    for genre in genres:
        all_sample_rates.extend(stats[genre]['sample_rates'])
        all_durations.extend(stats[genre]['durations'])
    
    if all_sample_rates:
        unique_rates = set(all_sample_rates)
        print(f"   Sample rates found: {unique_rates}")
        if len(unique_rates) > 1:
            print(f"   ℹ️  Multiple sample rates detected (will be resampled to 16kHz)")
        
        avg_duration = np.mean(all_durations)
        min_duration = np.min(all_durations)
        max_duration = np.max(all_durations)
        
        print(f"   Average duration: {avg_duration:.1f}s")
        print(f"   Duration range: {min_duration:.1f}s - {max_duration:.1f}s")
        
        if min_duration < 10:
            print(f"   ⚠️  Some files are very short (< 10s)")
    
    # List invalid files if any
    total_invalid = sum(stats[g]['invalid'] for g in genres)
    if total_invalid > 0:
        print(f"\n❌ Invalid Files ({total_invalid} total):")
        for genre in genres:
            if stats[genre]['errors']:
                print(f"\n   {genre}:")
                for filename, error in stats[genre]['errors'][:5]:  # Show first 5
                    print(f"      - {filename}: {error}")
                if len(stats[genre]['errors']) > 5:
                    print(f"      ... and {len(stats[genre]['errors']) - 5} more")
    
    # Training recommendations
    print(f"\n{'='*70}")
    print("TRAINING RECOMMENDATIONS")
    print("="*70)
    
    if ready_for_training and total_files >= 100:
        print("✅ Dataset is ready for training!")
        print("\nRecommended settings:")
        print(f"   DURATION = 10-15 seconds")
        print(f"   BATCH_SIZE = {min(32, total_files // 20)}")
        print(f"   EPOCHS = 50-100")
        print(f"   AUGMENTATION_FACTOR = {5 if total_files < 300 else 3}")
        
        with_augmentation = total_files * (5 if total_files < 300 else 3)
        print(f"\nWith augmentation: ~{with_augmentation} training samples")
        
    elif total_files < 100:
        print("⚠️  Dataset is too small for reliable training.")
        print(f"   Current: {total_files} files")
        print(f"   Recommended: 300+ files (100+ per genre)")
        print("\nConsider:")
        print("   - Collecting more data")
        print("   - Using aggressive augmentation")
        print("   - Transfer learning (already implemented)")
    else:
        print("⚠️  Dataset has issues that need to be addressed:")
        print("   - Ensure all genres have sufficient valid files")
        print("   - Remove or fix corrupted audio files")
        print("   - Balance the dataset across genres")
    
    print("\n" + "="*70)
    print("Validation complete! 🎉")
    print("="*70)

if __name__ == "__main__":
    main()
