"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  LIP READING DATA COLLECTOR v3.0+                         ║
║                Visual Komputer Cerdas Project - Lip Recognition           ║
║        Python 3.11.9 | MediaPipe Face Mesh (40 lip + teeth + corners)    ║
╚════════════════════════════════════════════════════════════════════════════╝

FEATURES v3.0:
✓ MediaPipe Face Mesh (40 detailed lip landmarks)
✓ Teeth boundary detection (8 landmarks) - untuk /s/, /z/, /f/, /v/
✓ Mouth corner tracking (2 landmarks) - untuk /p/, /m/ rounding
✓ Derived features: aperture, mouth_width, teeth_visibility, etc
✓ Real-time visualization with teeth + corners
✓ Multi-format export: .mp4 + .npy + .csv + .json
✓ Consonant-optimized: /p/, /b/, /m/, /f/, /v/, /s/, /z/ ready
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import csv
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

class Config:
    """Global configuration"""
    RECORD_DURATION = 10
    TARGET_FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    COUNTDOWN_SECS = 3
    DATA_ROOT = "dataset"
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # 40 Lip landmarks (outer + inner)
    LIP_LANDMARKS = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,    # Outer top (10)
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  # Outer bottom (10)
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,    # Inner top aperture (10)
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95    # Inner bottom aperture (10)
    ]
    
    # 8 Teeth boundaries (untuk detect consonants /s/, /z/, /f/, /v/)
    # Top teeth: 131, 130, 129, 128
    # Bottom teeth: 361, 360, 359, 358
    TEETH_LANDMARKS = [
        131, 130, 129, 128,      # Top teeth
        361, 360, 359, 358       # Bottom teeth
    ]
    
    # 2 Mouth corners (lip commissures) - untuk /p/, /m/ rounding
    MOUTH_CORNERS = [
        61, 291  # Left corner, Right corner
    ]
    
    # Inner lip aperture points (untuk aperture height calculation)
    APERTURE_TOP = [78, 191, 80, 81, 82]      # Top inner lip
    APERTURE_BOTTOM = [308, 324, 318, 402, 317]  # Bottom inner lip
    
    # Colors (BGR)
    COLOR_LIP_OUTER = (0, 255, 128)      # Green
    COLOR_LIP_INNER = (255, 0, 128)      # Magenta
    COLOR_TEETH = (200, 200, 255)        # Light red
    COLOR_CORNERS = (0, 200, 255)        # Orange
    COLOR_OUTLINE = (0, 200, 255)
    COLOR_TEXT_WHITE = (255, 255, 255)
    COLOR_RECORDING = (0, 0, 220)
    COLOR_OVERLAY = (20, 20, 20)


# ═════════════════════════════════════════════════════════════════════════════
#  DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_directories(label: str, session_id: Optional[str] = None) -> Dict[str, Path]:
    """Setup folder structure untuk recording session"""
    base = Path(Config.DATA_ROOT) / label
    video_dir = base / "videos"
    frames_dir = base / "frames"
    landmarks_dir = base / "landmarks"
    features_dir = base / "features"
    
    for d in [video_dir, frames_dir, landmarks_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    timestamp = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    paths = {
        "base": base,
        "video": video_dir / f"{label}_{timestamp}.mp4",
        "frames": frames_dir / timestamp,
        "landmarks_npy": landmarks_dir / f"{label}_{timestamp}.npy",
        "landmarks_csv": landmarks_dir / f"{label}_{timestamp}.csv",
        "features_csv": features_dir / f"{label}_{timestamp}_features.csv",
        "metadata": base / f"metadata_{timestamp}.json",
    }
    
    paths["frames"].mkdir(parents=True, exist_ok=True)
    return paths


# ═════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE PROCESSOR - ENHANCED
# ═════════════════════════════════════════════════════════════════════════════

class MediaPipeProcessor:
    """MediaPipe Face Mesh - 40 lip + 8 teeth + 2 corners + derived features"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        )
    
    def extract_derived_features(self, landmarks, h: int, w: int) -> Dict:
        """
        Extract derived features untuk consonant distinction:
        1. aperture_height - mouth opening (distinguish /æ/ vs /ə/)
        2. mouth_width - lip rounding (distinguish /p/ vs /b/)
        3. teeth_visibility - visibility ratio (/s/, /z/ have exposed teeth)
        4. mouth_corners_distance - lip compression (/p/, /m/ rounded)
        5. lip_protrusion - forward/backward (/f/, /v/ need contact)
        6. inner_outer_ratio - detail level
        7. top_aperture_width - top mouth opening
        8. bottom_aperture_width - bottom mouth opening
        9. vertical_asymmetry - jaw position
        """
        
        features = {}
        
        # Get pixel coordinates untuk calculations
        aperture_top_pts = []
        aperture_bottom_pts = []
        teeth_pts = []
        corner_pts = []
        
        for idx in Config.APERTURE_TOP:
            landmark = landmarks.landmark[idx]
            aperture_top_pts.append([landmark.x * w, landmark.y * h])
        
        for idx in Config.APERTURE_BOTTOM:
            landmark = landmarks.landmark[idx]
            aperture_bottom_pts.append([landmark.x * w, landmark.y * h])
        
        for idx in Config.TEETH_LANDMARKS:
            landmark = landmarks.landmark[idx]
            teeth_pts.append([landmark.x * w, landmark.y * h])
        
        for idx in Config.MOUTH_CORNERS:
            landmark = landmarks.landmark[idx]
            corner_pts.append([landmark.x * w, landmark.y * h])
        
        aperture_top_pts = np.array(aperture_top_pts)
        aperture_bottom_pts = np.array(aperture_bottom_pts)
        teeth_pts = np.array(teeth_pts)
        corner_pts = np.array(corner_pts)
        
        # 1. APERTURE HEIGHT (vertical distance between inner lips)
        if len(aperture_top_pts) > 0 and len(aperture_bottom_pts) > 0:
            top_y = np.mean(aperture_top_pts[:, 1])
            bottom_y = np.mean(aperture_bottom_pts[:, 1])
            aperture_height = abs(bottom_y - top_y)
            features['aperture_height'] = float(aperture_height)
        else:
            features['aperture_height'] = 0.0
        
        # 2. MOUTH WIDTH (distance between left and right corners)
        if len(corner_pts) == 2:
            mouth_width = abs(corner_pts[1][0] - corner_pts[0][0])
            features['mouth_width'] = float(mouth_width)
        else:
            features['mouth_width'] = 0.0
        
        # 3. TEETH VISIBILITY RATIO
        # Distance from aperture to teeth (small = more visible, good for /s/, /z/)
        if len(teeth_pts) > 0 and len(aperture_bottom_pts) > 0:
            avg_teeth_y = np.mean(teeth_pts[:, 1])
            avg_aperture_bottom_y = np.mean(aperture_bottom_pts[:, 1])
            teeth_visibility = max(0, avg_aperture_bottom_y - avg_teeth_y)
            features['teeth_visibility'] = float(teeth_visibility)
        else:
            features['teeth_visibility'] = 0.0
        
        # 4. MOUTH CORNERS DISTANCE (lip compression degree)
        # For /p/, /m/ = lips more compressed (smaller distance)
        if len(corner_pts) == 2:
            # Normalize by mouth width
            corners_dist = np.linalg.norm(corner_pts[1] - corner_pts[0])
            features['corners_distance'] = float(corners_dist)
        else:
            features['corners_distance'] = 0.0
        
        # 5. LIP PROTRUSION (how forward lips are)
        # Using depth (z) component from MediaPipe
        if len(aperture_bottom_pts) > 0:
            protrusion_pts = []
            for idx in Config.APERTURE_BOTTOM:
                landmark = landmarks.landmark[idx]
                protrusion_pts.append(landmark.z)
            features['protrusion_z'] = float(np.mean(protrusion_pts))
        else:
            features['protrusion_z'] = 0.0
        
        # 6. INNER-OUTER LIP RATIO
        inner_pts_count = len(aperture_top_pts) + len(aperture_bottom_pts)
        outer_pts_count = 20  # Fixed
        features['lip_detail_ratio'] = float(inner_pts_count / (outer_pts_count + 1))
        
        # 7. TOP APERTURE WIDTH (width of upper inner aperture)
        if len(aperture_top_pts) > 1:
            top_width = abs(aperture_top_pts[-1][0] - aperture_top_pts[0][0])
            features['top_aperture_width'] = float(top_width)
        else:
            features['top_aperture_width'] = 0.0
        
        # 8. BOTTOM APERTURE WIDTH
        if len(aperture_bottom_pts) > 1:
            bottom_width = abs(aperture_bottom_pts[-1][0] - aperture_bottom_pts[0][0])
            features['bottom_aperture_width'] = float(bottom_width)
        else:
            features['bottom_aperture_width'] = 0.0
        
        # 9. VERTICAL ASYMMETRY (jaw tilt indicator)
        # Difference in lip heights at left vs right
        if len(aperture_top_pts) > 0:
            left_y = aperture_top_pts[0][1]
            right_y = aperture_top_pts[-1][1]
            asymmetry = abs(right_y - left_y)
            features['vertical_asymmetry'] = float(asymmetry)
        else:
            features['vertical_asymmetry'] = 0.0
        
        return features
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract 40 lip + 8 teeth + features dari frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        normalized_coords = []
        pixel_coords = []
        xs, ys = [], []
        
        # ===== LIP LANDMARKS (40) =====
        for idx in Config.LIP_LANDMARKS:
            landmark = landmarks.landmark[idx]
            
            normalized_coords.append({
                'idx': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'type': 'lip'
            })
            
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            pixel_coords.append({
                'idx': idx,
                'x': px,
                'y': py,
                'z': landmark.z,
                'type': 'lip'
            })
            
            xs.append(px)
            ys.append(py)
        
        # ===== TEETH LANDMARKS (8) =====
        teeth_coords_pixel = []
        for idx in Config.TEETH_LANDMARKS:
            landmark = landmarks.landmark[idx]
            
            normalized_coords.append({
                'idx': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'type': 'teeth'
            })
            
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            pixel_coords.append({
                'idx': idx,
                'x': px,
                'y': py,
                'z': landmark.z,
                'type': 'teeth'
            })
            
            teeth_coords_pixel.append((px, py))
            xs.append(px)
            ys.append(py)
        
        # ===== MOUTH CORNERS (2) =====
        corners_pixel = []
        for idx in Config.MOUTH_CORNERS:
            landmark = landmarks.landmark[idx]
            
            normalized_coords.append({
                'idx': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'type': 'corner'
            })
            
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            pixel_coords.append({
                'idx': idx,
                'x': px,
                'y': py,
                'z': landmark.z,
                'type': 'corner'
            })
            
            corners_pixel.append((px, py))
            xs.append(px)
            ys.append(py)
        
        # Calculate ROI bounding box
        padding = 30
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(w, max(xs) + padding)
        y_max = min(h, max(ys) + padding)
        
        roi_box = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min,
        }
        
        # Extract derived features
        derived_features = self.extract_derived_features(landmarks, h, w)
        
        return {
            'normalized': normalized_coords,
            'pixel': pixel_coords,
            'roi_box': roi_box,
            'derived_features': derived_features,
            'timestamp': datetime.now().isoformat(),
            'teeth_coords': teeth_coords_pixel,
            'corners_coords': corners_pixel,
        }
    
    def close(self):
        if self.face_mesh:
            self.face_mesh.close()


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION - ENHANCED
# ═════════════════════════════════════════════════════════════════════════════

def draw_landmarks(frame: np.ndarray, landmark_data: Dict) -> np.ndarray:
    """Draw 40 lip + 8 teeth + 2 corners + ROI box"""
    if landmark_data is None:
        return frame
    
    roi_box = landmark_data['roi_box']
    pixel_coords = landmark_data['pixel']
    
    # Draw landmark points dengan color coding
    for coord in pixel_coords:
        x, y = coord['x'], coord['y']
        coord_type = coord.get('type', 'lip')
        
        if coord_type == 'lip':
            # Distinguish outer vs inner lip
            idx = coord['idx']
            if idx in Config.LIP_LANDMARKS[:20]:
                color = Config.COLOR_LIP_OUTER
                radius = 3
            else:
                color = Config.COLOR_LIP_INNER
                radius = 2
        elif coord_type == 'teeth':
            color = Config.COLOR_TEETH
            radius = 3
        elif coord_type == 'corner':
            color = Config.COLOR_CORNERS
            radius = 4
        else:
            color = (255, 255, 255)
            radius = 2
        
        cv2.circle(frame, (x, y), radius, color, -1)
    
    # Draw ROI box
    cv2.rectangle(
        frame,
        (roi_box['x_min'], roi_box['y_min']),
        (roi_box['x_max'], roi_box['y_max']),
        Config.COLOR_OUTLINE, 2
    )
    
    # Draw teeth boundary line
    if len(landmark_data['teeth_coords']) >= 2:
        teeth = landmark_data['teeth_coords']
        for i in range(len(teeth) - 1):
            cv2.line(frame, teeth[i], teeth[i+1], Config.COLOR_TEETH, 2)
    
    # Draw mouth corner connection
    if len(landmark_data['corners_coords']) == 2:
        c1, c2 = landmark_data['corners_coords']
        cv2.line(frame, c1, c2, Config.COLOR_CORNERS, 2)
    
    return frame


def draw_hud(frame: np.ndarray, state: str, label: str, elapsed: float,
             frame_count: int, landmark_detected: bool, features: Optional[Dict] = None) -> np.ndarray:
    """Draw HUD panel dengan features info"""
    h, w = frame.shape[:2]
    
    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), Config.COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Label
    cv2.putText(frame, f"Kata: [{label.upper()}]",
                (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                Config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    
    # Detection status
    if landmark_detected:
        det_text = "✓ 40 LIP + 8 TEETH + FEATURES"
        det_color = Config.COLOR_LIP_OUTER
    else:
        det_text = "⚠ NO DETECTION"
        det_color = (0, 100, 255)
    
    cv2.putText(frame, det_text,
                (w - 360, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                det_color, 1, cv2.LINE_AA)
    
    # Show aperture height (key metric)
    if features and 'aperture_height' in features:
        aperture = features['aperture_height']
        mouth_width = features.get('mouth_width', 0)
        cv2.putText(frame, f"Aperture: {aperture:.0f}px | Width: {mouth_width:.0f}px",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    Config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    
    if state == "countdown":
        remaining = Config.COUNTDOWN_SECS - int(elapsed)
        text = str(max(0, remaining))
        color = (0, 255, 100) if remaining > 0 else (0, 255, 0)
        font_size = 3.0
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_size, 4)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_size, color, 4, cv2.LINE_AA)
    
    elif state == "recording":
        remaining = Config.RECORD_DURATION - elapsed
        progress = elapsed / Config.RECORD_DURATION
        
        # REC indicator
        if int(elapsed * 2) % 2 == 0:
            cv2.circle(frame, (w - 30, h - 30), 12, Config.COLOR_RECORDING, -1)
            cv2.putText(frame, "REC",
                        (w - 65, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        Config.COLOR_RECORDING, 2, cv2.LINE_AA)
        
        # Progress bar
        bar_x, bar_y = 10, h - 25
        bar_w, bar_h = w - 20, 14
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h),
                      (0, 210, 80), -1)
        
        cv2.putText(frame, f"{remaining:.1f}s | Frame: {frame_count}",
                    (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    Config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    
    return frame


# ═════════════════════════════════════════════════════════════════════════════
#  VIDEO RECORDING
# ═════════════════════════════════════════════════════════════════════════════

class VideoRecorder:
    """Handle video writing"""
    
    def __init__(self, output_path: Path, fps: int, width: int, height: int):
        codecs = ['mp4v', 'avc1', 'MJPG', 'XVID']
        self.writer = None
        self.output_path = output_path
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (width, height)
                )
                if writer.isOpened():
                    self.writer = writer
                    print(f"  ✓ Video writer codec: {codec}")
                    break
                writer.release()
            except:
                continue
        
        if self.writer is None:
            raise RuntimeError("Cannot initialize video writer!")
    
    def write_frame(self, frame: np.ndarray) -> bool:
        if self.writer is None:
            return False
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            print(f"  ⚠ Error writing frame: {e}")
            return False
    
    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None


# ═════════════════════════════════════════════════════════════════════════════
#  LANDMARK SAVING - ENHANCED
# ═════════════════════════════════════════════════════════════════════════════

def save_landmarks_data(all_landmarks: List[Dict], all_features: List[Dict], 
                       paths: Dict, label: str) -> None:
    """Save landmarks + features ke .npy dan .csv"""
    if not all_landmarks:
        print("  ⚠ Tidak ada landmark.")
        return
    
    # ===== SAVE LANDMARKS (.npy) =====
    landmarks_array = []
    for frame_data in all_landmarks:
        frame_coords = []
        for point in frame_data['normalized']:
            frame_coords.append([point['x'], point['y'], point['z']])
        landmarks_array.append(frame_coords)
    
    landmarks_array = np.array(landmarks_array, dtype=np.float32)
    
    np.save(str(paths['landmarks_npy']), landmarks_array)
    print(f"  ✓ Landmarks .npy: {paths['landmarks_npy']}")
    print(f"    Shape: {landmarks_array.shape} (frames × 50 points × 3 coords)")
    print(f"    → 40 lip + 8 teeth + 2 corners")
    
    # ===== SAVE DETAILED LANDMARKS (.csv) =====
    with open(paths['landmarks_csv'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'frame_idx', 'point_idx', 'landmark_id', 'type',
                         'x_norm', 'y_norm', 'z_norm', 'x_pixel', 'y_pixel'])
        
        for frame_idx, frame_data in enumerate(all_landmarks):
            norm_points = frame_data['normalized']
            pixel_points = frame_data['pixel']
            
            for point_idx, (norm_pt, pixel_pt) in enumerate(zip(norm_points, pixel_points)):
                writer.writerow([
                    label, frame_idx, point_idx,
                    norm_pt['idx'],
                    norm_pt.get('type', 'lip'),
                    norm_pt['x'], norm_pt['y'], norm_pt['z'],
                    pixel_pt['x'], pixel_pt['y']
                ])
    
    print(f"  ✓ Landmarks .csv: {paths['landmarks_csv']}")
    
    # ===== SAVE DERIVED FEATURES (.csv) =====
    if all_features:
        with open(paths['features_csv'], 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['frame_idx', 'aperture_height', 'mouth_width', 
                         'teeth_visibility', 'corners_distance', 'protrusion_z',
                         'lip_detail_ratio', 'top_aperture_width', 
                         'bottom_aperture_width', 'vertical_asymmetry']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_idx, feat in enumerate(all_features):
                feat_row = {'frame_idx': frame_idx}
                feat_row.update(feat)
                writer.writerow(feat_row)
        
        print(f"  ✓ Features .csv: {paths['features_csv']}")
        print(f"    → 9 derived metrics untuk consonant distinction")


def save_frame_with_landmarks(frame: np.ndarray, frame_idx: int,
                             frames_dir: Path, landmark_data: Optional[Dict]) -> Path:
    """Save frame dengan landmark visualization"""
    if landmark_data is not None:
        frame = draw_landmarks(frame, landmark_data)
    
    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(frame_path), frame)
    return frame_path


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN RECORDING SESSION
# ═════════════════════════════════════════════════════════════════════════════

def record_session(label: str, session_id: Optional[str] = None) -> Optional[Dict]:
    """Main recording loop"""
    
    paths = setup_directories(label, session_id)
    
    print(f"\n{'='*70}")
    print(f"📹 RECORDING: {label.upper()}")
    print(f"{'='*70}")
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        print("   Solutions:")
        print("   1. Check if camera connected")
        print("   2. sudo chown $USER /dev/video*")
        return None
    
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or Config.TARGET_FPS
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📷 Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
    print(f"   Duration: {Config.RECORD_DURATION}s")
    print(f"   Expected frames: {actual_fps * Config.RECORD_DURATION}")
    print(f"   Extracting: 40 lip + 8 teeth + 9 features")
    print(f"   Press [Q] to cancel\n")
    
    # Init video writer
    try:
        video_writer = VideoRecorder(paths['video'], actual_fps,
                                     actual_width, actual_height)
    except Exception as e:
        print(f"❌ {e}")
        cap.release()
        return None
    
    # Init MediaPipe
    processor = MediaPipeProcessor()
    
    all_landmarks = []
    all_features = []
    all_frames = []
    frame_count = 0
    landmark_count = 0
    
    print(f"▶ Countdown...\n")
    
    state = "countdown"
    start_time = time.time()
    recording_start = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process landmarks
            landmark_data = processor.process_frame(frame)
            landmark_detected = landmark_data is not None
            
            elapsed = time.time() - start_time
            
            # STATE MACHINE
            if state == "countdown":
                frame = draw_hud(frame, "countdown", label, elapsed, frame_count,
                               landmark_detected)
                
                if elapsed >= Config.COUNTDOWN_SECS:
                    state = "recording"
                    recording_start = time.time()
                    print("▶ RECORDING...\n")
            
            elif state == "recording":
                rec_elapsed = time.time() - recording_start
                frame = draw_hud(frame, "recording", label, rec_elapsed, frame_count,
                               landmark_detected,
                               landmark_data.get('derived_features') if landmark_data else None)
                
                # Write video
                if not video_writer.write_frame(frame):
                    print("❌ Write error")
                    break
                
                # Save frame
                frame_path = save_frame_with_landmarks(frame, frame_count,
                                                      paths['frames'], landmark_data)
                all_frames.append(str(frame_path))
                
                # Store landmark + features
                if landmark_data is not None:
                    all_landmarks.append(landmark_data)
                    all_features.append(landmark_data.get('derived_features', {}))
                    landmark_count += 1
                
                frame_count += 1
                
                # Check done
                if rec_elapsed >= Config.RECORD_DURATION:
                    print(f"\n✓ Done - {frame_count} frames\n")
                    break
            
            # Display
            cv2.imshow("Lip Reading Collector v3", frame)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n❌ Cancelled")
                cap.release()
                video_writer.release()
                cv2.destroyAllWindows()
                processor.close()
                return None
    
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        processor.close()
    
    # SAVE DATA
    print(f"💾 Saving...\n")
    save_landmarks_data(all_landmarks, all_features, paths, label)
    
    detection_rate = (landmark_count / frame_count * 100) if frame_count > 0 else 0
    
    print(f"\n📊 Statistics:")
    print(f"   Total frames: {frame_count}")
    print(f"   Detected: {landmark_count}")
    print(f"   Detection rate: {detection_rate:.1f}%")
    
    # METADATA
    metadata = {
        'label': label,
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        'recorded_frames': frame_count,
        'detected_frames': landmark_count,
        'detection_rate': detection_rate,
        'fps': actual_fps,
        'resolution': [actual_width, actual_height],
        'duration': Config.RECORD_DURATION,
        'video_path': str(paths['video']),
        'frames_path': str(paths['frames']),
        'landmarks_npy': str(paths['landmarks_npy']),
        'landmarks_csv': str(paths['landmarks_csv']),
        'features_csv': str(paths['features_csv']),
        'total_landmarks': 50,
        'landmark_breakdown': {
            'lip_outer': 20,
            'lip_inner_aperture': 20,
            'teeth': 8,
            'mouth_corners': 2,
        },
        'derived_features': 9,
        'feature_list': [
            'aperture_height',
            'mouth_width',
            'teeth_visibility',
            'corners_distance',
            'protrusion_z',
            'lip_detail_ratio',
            'top_aperture_width',
            'bottom_aperture_width',
            'vertical_asymmetry'
        ],
        'consonant_support': {
            'visible_consonants': ['/p/', '/b/', '/m/', '/f/', '/v/', '/s/', '/z/'],
            'partial_consonants': ['/t/', '/d/ (indirect from jaw)'],
            'features_explanation': {
                'aperture_height': 'Distinguish /æ/ vs /ə/ vowels, opens for plosives',
                'mouth_width': 'Lip rounding for /p/, /m/, /w/',
                'teeth_visibility': 'Fricatives /s/, /z/, /f/ expose teeth',
                'corners_distance': 'Lip compression for /p/, /m/',
                'top_aperture_width': 'Upper mouth opening for consonant articulation',
                'bottom_aperture_width': 'Lower mouth opening variation',
                'vertical_asymmetry': 'Jaw tilt during /t/ vs /d/'
            }
        },
        'frames': all_frames,
    }
    
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Metadata: {paths['metadata']}")
    
    print(f"\n{'='*70}")
    print(f"✅ SESSION COMPLETE - v3.0 Enhanced")
    print(f"{'='*70}\n")
    
    return metadata


# ═════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ═════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "═" * 72)
    print("  🎤  LIP READING DATA COLLECTOR v3.0+")
    print("  Visual Komputer Cerdas Project | Python 3.11.9")
    print("  40 Lip + 8 Teeth + 9 Features (Consonant Optimized)")
    print("═" * 72)


def main():
    print_banner()
    
    while True:
        print("\n" + "─" * 72)
        label = input("  📝 Masukkan KATA (contoh: aku, halo, kamu, test)\n"
                     "     Atau [exit] untuk keluar: ").strip().lower()
        
        if label in ("exit", "quit", "q", ""):
            print("\n  👋 Sampai jumpa!\n")
            break
        
        if not label.isalpha():
            print("  ⚠ Hanya huruf")
            continue
        
        print(f"\n  ℹ Setting:")
        print(f"     • Duration: {Config.RECORD_DURATION}s")
        print(f"     • FPS: {Config.TARGET_FPS}")
        print(f"     • Landmarks: 50 (40 lip + 8 teeth + 2 corners)")
        print(f"     • Derived features: 9 (consonant optimized)")
        print(f"     • Output: video + frames + landmarks + features")
        
        while True:
            choice = input(f"\n  Mulai '{label.upper()}'? [y/n/back]: ").strip().lower()
            
            if choice in ("n", "no", "back"):
                break
            
            if choice in ("y", "yes", ""):
                metadata = record_session(label)
                
                if metadata:
                    another = input("\n  Rekam lagi? [y/n]: ").strip().lower()
                    if another not in ("y", "yes"):
                        break
            else:
                print("  Ketik 'y', 'n', atau 'back'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)