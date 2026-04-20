# 🎤 Indonesian Lip Reading - Visual Komputer Cerdas Project

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![MediaPipe](https://img.shields.io/badge/mediapipe-latest-green)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/opencv-python-red)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Real-time lip reading system untuk mengenali kata-kata Indonesia dari gerakan bibir menggunakan 3D CNN dan MediaPipe Face Mesh landmarks.

## 🎯 Project Overview

Proyek ini bertujuan untuk **membaca dan mengenali kata-kata Indonesia** dari gerakan bibir (lip reading) menggunakan:
- **40 Lip Landmarks** (MediaPipe Face Mesh) untuk detail articulation
- **8 Teeth Landmarks** untuk fricative consonants (/s/, /z/, /f/, /v/)
- **9 Derived Features** untuk membedakan consonant pronunciation
- **3D CNN Model** untuk temporal analysis gerakan bibir

### Target Consonants (Terlihat dari Lips):
✅ **/p/, /b/, /m/** - Bilabials (closes both lips)
✅ **/f/, /v/** - Labiodentals (lower lip + teeth)
✅ **/s/, /z/** - Alveolar fricatives (exposed teeth)
⚠️ **/t/, /d/, /n/** - Alveolar (partial, indirect)
⚠️ **/k/, /g/** - Velars (mostly internal, hidden)

---

## 📋 Features

### v3.0 - Enhanced Consonant Detection
- ✅ **40 Lip Landmarks**: outer contour (20) + inner aperture (20)
- ✅ **8 Teeth Landmarks**: top (4) + bottom (4) untuk fricative detection
- ✅ **2 Mouth Corners**: commissures untuk lip rounding (/p/, /m/)
- ✅ **9 Derived Metrics**:
  - `aperture_height` - mouth opening
  - `mouth_width` - lip spreading
  - `teeth_visibility` - fricative indicator
  - `corners_distance` - lip compression
  - `protrusion_z` - forward/backward position
  - `lip_detail_ratio` - feature dimensionality
  - `top_aperture_width` - upper mouth opening
  - `bottom_aperture_width` - lower mouth opening
  - `vertical_asymmetry` - jaw tilt
- ✅ **Real-time Visualization**: color-coded landmarks (green=outer, magenta=inner, red=teeth, orange=corners)
- ✅ **Multi-format Export**: .mp4 video + .npy arrays + .csv metrics + .json metadata
- ✅ **Modular Architecture**: Easy to extend dan modify

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd "building-model"

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

**Requirements:**
- Python 3.11.9
- OpenCV (cv2)
- MediaPipe
- NumPy
- TensorFlow/Keras (untuk training)

### Run Data Collection

```python
# Single sample
from DataCollect import record_session

metadata = record_session(label='halo')
# Output: Video + frames + landmarks + features

# Batch collection (recommended)
from DataCollect import batch_collect_data

config = [
    {"kata": "halo", "speaker": "fahri", "count": 5},
    {"kata": "halo", "speaker": "budi", "count": 3},
    {"kata": "terima_kasih", "speaker": "fahri", "count": 5},
]

results = batch_collect_data(config)
```

---

## 📁 Project Structure

```
building-model/
├── DataCollect.py                    # ⭐ Updated collector v3.0 (40 lip + 8 teeth + 9 features)
├── data_collecting.ipynb             # Legacy collector (Cascade - jangan pakai)
├── model.ipynb                       # 3D CNN training notebook
├── collect.ipynb                     # Helper functions
│
├── dataset/                          # Generated during collection
│   ├── halo/
│   │   ├── speaker_fahri/
│   │   │   ├── videos/
│   │   │   │   └── halo_fahri_YYYYMMDD_HHMMSS.mp4
│   │   │   ├── frames/
│   │   │   │   └── YYYYMMDD_HHMMSS/
│   │   │   │       └── frame_000000.jpg (with landmarks overlay)
│   │   │   ├── landmarks/
│   │   │   │   ├── halo_fahri_YYYYMMDD_HHMMSS.npy (50×3)
│   │   │   │   └── halo_fahri_YYYYMMDD_HHMMSS.csv
│   │   │   ├── features/
│   │   │   │   └── halo_fahri_YYYYMMDD_HHMMSS_features.csv (9 metrics)
│   │   │   └── metadata_YYYYMMDD_HHMMSS.json
│   │   └── speaker_budi/ ...
│   ├── terima_kasih/ ...
│   └── ...
│
├── models/
│   ├── face_landmarker.task          # MediaPipe model
│   └── lip_reading_model.h5          # Trained 3D CNN
│
├── IDLRW-DATASET/                    # External reference dataset
├── requirements.txt                  # Core dependencies
├── requirements-ml.txt               # ML training libraries
└── README.md                         # This file
```

---

## 💻 Usage Examples

### 1. Collect Single Recording

```python
from DataCollect import record_session

# Interactive CLI
metadata = record_session(label='halo')  # Prompts untuk input kata
```

### 2. Collect Multiple Words & Speakers

```python
from DataCollect import batch_collect_data, get_dataset_statistics

# Define collection plan
config = [
    {"kata": "aku", "speaker": "fahri", "count": 5},
    {"kata": "aku", "speaker": "budi", "count": 3},
    {"kata": "kamu", "speaker": "fahri", "count": 5},
    {"kata": "kamu", "speaker": "budi", "count": 3},
]

# Collect all
results = batch_collect_data(config)

# Monitor progress
stats = get_dataset_statistics()
```

### 3. Data Quality Check

```python
from DataCollect import check_data_quality, cleanup_bad_samples

# Find issues
check_data_quality()

# Remove low quality samples (< 50% detection rate)
cleanup_bad_samples(min_detection_rate=0.5)
```

### 4. Access Collected Data

```python
import numpy as np
import pandas as pd

# Load landmarks array
landmarks = np.load('dataset/halo/landmarks/halo_fahri_*.npy')
# Shape: (n_frames, 50, 3)
# 50 = 40 lip + 8 teeth + 2 corners
# 3 = x, y, z coordinates

# Load features CSV
features_df = pd.read_csv('dataset/halo/features/halo_fahri_*_features.csv')
# Columns: frame_idx, aperture_height, mouth_width, teeth_visibility, ...

# Load landmarks CSV (detailed)
landmarks_df = pd.read_csv('dataset/halo/landmarks/halo_fahri_*.csv')
# Columns: label, frame_idx, point_idx, landmark_id, type, x_norm, y_norm, z_norm, x_pixel, y_pixel
```

---

## 📊 Data Collection Guide

### Hardware Requirements
- 📷 **Webcam**: 30fps minimum (USB 2.0+ recommended)
- 💻 **CPU**: Intel i5/AMD Ryzen 5 or better
- 🖥️ **RAM**: 8GB minimum
- 📍 **OS**: Linux (primary), macOS (supported), Windows (partial)

### Optimal Recording Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Duration** | 20s | 200 frame @ 10fps |
| **FPS** | 30 (target) | Actual depends on camera |
| **Resolution** | 640×480 | Minimum for mouth detail |
| **Lighting** | Natural or consistent | Avoid shadows on mouth |
| **Distance** | 30-60cm | Face centered, clear mouth |
| **Background** | Plain | Reduces detection artifacts |
| **Mouth Detection Target** | >70% | Acceptable for training |

### Collection Workflow

```python
# Step 1: Diagnostic - cek FPS kamera aktual
from DataCollect import diagnose_camera_settings
actual_fps, frames, elapsed = diagnose_camera_settings()

# Step 2: Collect dengan durasi yang tepat
metadata = record_session(label='halo')

# Step 3: Check hasil
stats = get_dataset_statistics()
check_data_quality()
```

---

## 🎥 Video Recording Tips

### ✅ Good Practices
- ✓ **Clear Speech**: Speak naturally with clear pronunciation
- ✓ **Face Centered**: Keep mouth in center of frame
- ✓ **Good Lighting**: Natural light from front/side (no backlight)
- ✓ **Consistent Pace**: Vary speed slightly (normal, slow, fast)
- ✓ **Multiple Takes**: Different angles/expressions
- ✓ **Clean Background**: Reduce visual distractions

### ❌ Avoid
- ✗ Backlighting atau side shadows
- ✗ Dimly lit environment
- ✗ Too close atau too far from camera
- ✗ Exaggerated mouth movements
- ✗ Hands covering mouth
- ✗ Sunglasses atau face obstruction

---

## 📈 Data Export Formats

### 1. **Video (.mp4/.avi)**
Raw recording dengan landmark visualization (HUD overlay)
```
video_path: dataset/halo/videos/halo_fahri_YYYYMMDD_HHMMSS.mp4
Size: ~20-50 MB per 20s recording
```

### 2. **Frames (.jpg)**
Individual frames dengan landmark boxes drawn
```
frames_path: dataset/halo/frames/YYYYMMDD_HHMMSS/frame_000000.jpg
Each: ~20-30 KB | Total: 200-300 frames per recording
```

### 3. **Landmarks Array (.npy)** ⭐ FOR ML
NumPy binary format - **direct input untuk 3D CNN**
```python
landmarks = np.load('dataset/halo/landmarks/halo_fahri_YYYYMMDD_HHMMSS.npy')
# Shape: (n_frames, 50, 3)
# n_frames: 200-300 (depending on FPS)
# 50 points: 40 lip + 8 teeth + 2 corners
# 3 coords: x, y, z (normalized 0-1)
```

### 4. **Landmarks Table (.csv)**
Detailed CSV dengan semua coordinates
```
Columns: label, frame_idx, point_idx, landmark_id, type, 
         x_norm, y_norm, z_norm, x_pixel, y_pixel
Rows: n_frames × 50 points
```

### 5. **Derived Features (.csv)** ⭐ FOR ANALYSIS
9 metrics untuk consonant distinction
```
Columns: frame_idx, aperture_height, mouth_width, teeth_visibility,
         corners_distance, protrusion_z, lip_detail_ratio,
         top_aperture_width, bottom_aperture_width, vertical_asymmetry
Rows: n_frames
```

### 6. **Metadata (.json)**
Recording session metadata + statistics
```json
{
  "kata": "halo",
  "speaker": "fahri",
  "timestamp": "2026-04-20T12:00:00",
  "recorded_frames": 300,
  "detected_frames": 280,
  "detection_rate": 93.3,
  "fps": 30,
  "resolution": [640, 480],
  "landmark_breakdown": {
    "lip_outer": 20,
    "lip_inner_aperture": 20,
    "teeth": 8,
    "mouth_corners": 2
  }
}
```

---

## 🔍 Troubleshooting

### Problem: No Frames Extracted
**Cause**: Camera not accessible or recording failed
```bash
# Fix permissions
sudo chown $USER /dev/video*

# Close other apps using camera
# Try different camera index: VideoCapture(1), VideoCapture(2)
```

### Problem: Low Detection Rate (<50%)
**Cause**: Poor lighting or unfavorable camera angle
```python
# Solution 1: Improve lighting
# - Move to natural light window
# - Use consistent desk lamp
# - Avoid harsh shadows

# Solution 2: Adjust camera angle
# - Center mouth in frame
# - Slight downward angle
# - Move 30-50cm from camera
```

### Problem: Very Few Frames (~100 in 10s)
**Cause**: MediaPipe inference is slow (actual FPS < target FPS)
```python
# Solution 1: Increase duration (easiest)
metadata = record_session(label='halo')  # duration=20 instead of 10

# Solution 2: Lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

---

## 📚 Model Training

Setelah data collection selesai, train 3D CNN model:

```python
# See model.ipynb untuk training pipeline
# Input: .npy files dari dataset/
# Output: Trained model.h5

# Expected accuracy:
# - Single consonants: 85-92%
# - Word-level: 75-85%
```

---

## 📊 Expected Dataset Size

Recommended collection config:

```python
config = [
    {"kata": "aku", "speaker": "fahri", "count": 5},
    {"kata": "aku", "speaker": "citra", "count": 3},
    {"kata": "kamu", "speaker": "fahri", "count": 5},
    {"kata": "kamu", "speaker": "citra", "count": 3},
]

# Expected:
# - Total samples: 16
# - Total frames: ~3,200 (16 × 200)
# - Storage (videos + frames): ~350 MB
# - Storage (.npy only): ~30 MB
```

---

## 🛠️ Development

### Version History
- **v3.0** (Current) - Enhanced with teeth + features
  - 40 lip + 8 teeth + 2 corners landmarks
  - 9 derived metrics
  - Better consonant detection
- **v2.0** - Initial MediaPipe version (40 lip landmarks)
- **v1.0** - Cascade Classifier (deprecated, 5 landmarks only)

### Next Features (Roadmap)
- [ ] Audio-visual fusion (combine lips + audio)
- [ ] Multi-language support
- [ ] Real-time inference GUI
- [ ] Edge device optimization (TFLite)
- [ ] Continuous speech recognition

---

## 📄 License

MIT License - See LICENSE file

---

## 👥 Contributing

Contributions welcome! Areas:
- Data collection (more speakers, words)
- Model optimization
- Platform support (Windows native, edge devices)
- Documentation improvements

---

## 📞 Support

Pertanyaan? Refer ke:
1. **Troubleshooting section** di atas
2. **Code comments** di DataCollect.py
3. **Jupyter notebooks** untuk examples

---

## 🙏 Acknowledgements

Thanks to:
- **MediaPipe** - Face mesh landmark detection
- **OpenCV** - Video processing
- **TensorFlow/Keras** - Deep learning framework

---

**Happy lip reading! 🎤🎯**

*Last Updated: April 20, 2026*