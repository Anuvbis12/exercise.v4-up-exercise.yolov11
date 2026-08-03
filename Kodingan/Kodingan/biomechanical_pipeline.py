import os
import time
import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO

class KeypointEMASmoother:
    """
    Exponential Moving Average (EMA) Smoother untuk memuluskan pergerakan keypoint
    tanpa menimbulkan lag temporal yang parah.
    """
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.smoothed_kpts = None

    def update(self, keypoints):
        if self.smoothed_kpts is None or self.smoothed_kpts.shape != keypoints.shape:
            self.smoothed_kpts = keypoints.copy()
        else:
            self.smoothed_kpts = self.alpha * keypoints + (1 - self.alpha) * self.smoothed_kpts
        return self.smoothed_kpts

    def reset(self):
        self.smoothed_kpts = None


class BiomechanicalPipeline:
    def __init__(self, pose_model_path, classifier_model_path=None, conf_threshold=0.5):
        # Auto-detect GPU/CPU device
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        # Load Model Deteksi (Support .pt dan .engine TensorRT)
        print(f"📦 Loading Model Detector dari: {pose_model_path}")
        self.detector = YOLO(pose_model_path)
        self.conf_threshold = conf_threshold
        
        # Load Classifier (jika file .pkl / .joblib ada)
        self.classifier = None
        if classifier_model_path and os.path.exists(classifier_model_path):
            try:
                import joblib
                self.classifier = joblib.load(classifier_model_path)
                print(f"✅ Classifier berhasil dimuat dari: {classifier_model_path}")
            except Exception as e:
                print(f"⚠️ Gagal memuat classifier: {e}")
                
        # Inisialisasi Smoother EMA
        self.smoother = KeypointEMASmoother(alpha=0.65)
        
        # Metrik Performa Real-Time
        self.prev_time = time.time()
        self.fps = 0.0

    def stage1_extract_and_smooth(self, frame):
        """
        Stage 1: Ekstraksi 17 keypoint COCO dan temporal smoothing teroptimasi.
        """
        # Eksekusi inferensi di GPU dengan FP16 (jika didukung)
        results = self.detector.predict(
            frame, 
            device=self.device,
            half=(self.device != 'cpu'),
            verbose=False
        )
        
        if not results or len(results[0].keypoints) == 0:
            self.smoother.reset()
            return None, None
            
        kpt_data = results[0].keypoints[0] # Ambil deteksi pose orang pertama
        keypoints = kpt_data.xy[0].cpu().numpy() # Shape (17, 2)
        confidences = kpt_data.conf[0].cpu().numpy() if kpt_data.conf is not None else np.ones(17)
        
        # Terapkan EMA Smoothing pada koordinat keypoint
        smoothed_keypoints = self.smoother.update(keypoints)
        return smoothed_keypoints, confidences

    def stage2_classify_movement(self, keypoints):
        """
        Stage 2: Klasifikasi jenis gerakan sekuensial.
        """
        if self.classifier is not None:
            features = keypoints.flatten().reshape(1, -1)
            movement_class = self.classifier.predict(features)[0]
            return movement_class
        
        # Default fallback jika classifier belum di-train
        return "Push-up"

    def stage3_planar_homography(self, keypoints, confidences):
        """
        Stage 3a: Mengoreksi distorsi kamera miring menggunakan RANSAC Homography.
        Memetakan dari ruang citra miring ke tampilan samping (canonical side-view).
        """
        # Indeks COCO: 5=Bahu Kiri, 6=Bahu Kanan, 11=Pinggul Kiri, 12=Pinggul Kanan
        ref_indices = [5, 6, 11, 12]
        
        # Cek apakah keypoint acuan memiliki confidence yang memadai
        if any(confidences[idx] < self.conf_threshold for idx in ref_indices):
            return keypoints

        src_pts = np.array([keypoints[idx] for idx in ref_indices], dtype=np.float32)
        dst_pts = np.array([[0, 0], [100, 0], [0, 200], [100, 200]], dtype=np.float32)
        
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            kpts_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
            transformed = (H @ kpts_homogeneous.T).T
            
            # Cegah pembagian dengan nol
            denom = np.where(transformed[:, 2:] == 0, 1e-6, transformed[:, 2:])
            transformed = transformed[:, :2] / denom
            return transformed
        return keypoints

    def calculate_angle(self, A, B, C):
        """
        Menghitung sudut 2D di titik sendi B menggunakan Hukum Kosinus.
        """
        a = math.dist(B, C)
        c = math.dist(A, B)
        b = math.dist(A, C)
        
        if a == 0 or c == 0:
            return 0.0
            
        cos_val = (a**2 + c**2 - b**2) / (2 * a * c)
        angle = math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0)))
        return angle

    def stage3_biomechanical_jaa(self, movement, keypoints, confidences, frame):
        """
        Stage 3b: Evaluasi postur berdasarkan JAA (Joint Angle Accuracy).
        """
        status = "TERDETEKSI"
        color = (0, 255, 255)
        
        if movement == "Push-up":
            # Indeks: Bahu(5), Siku(7), Pergelangan Tangan(9), Pinggul(11), Pergelangan Kaki(15)
            if confidences[5] > self.conf_threshold and confidences[11] > self.conf_threshold and confidences[15] > self.conf_threshold:
                hip_angle = self.calculate_angle(keypoints[5], keypoints[11], keypoints[15])
                
                if hip_angle > 165:
                    status = "POSTUR SEMPURNA"
                    color = (0, 255, 0)
                elif hip_angle > 150:
                    status = "POSTUR BAIK"
                    color = (0, 255, 255)
                else:
                    status = "POSTUR BURUK (Pinggul Terlalu Turun!)"
                    color = (0, 0, 255)

        # Update FPS Calculation
        curr_time = time.time()
        self.fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time

        # Visualisasi Overlay Real-Time
        cv2.putText(frame, f"Gerakan: {movement}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def process_frame(self, frame):
        # Tahap 1: Ekstraksi & EMA Smoothing
        keypoints, confidences = self.stage1_extract_and_smooth(frame)
        if keypoints is None: 
            return frame
        
        # Tahap 2: Klasifikasi Gerakan
        movement = self.stage2_classify_movement(keypoints)
        
        # Tahap 3: Koreksi Homografi & Evaluasi Biomekanik
        normalized_kpts = self.stage3_planar_homography(keypoints, confidences)
        annotated_frame = self.stage3_biomechanical_jaa(movement, normalized_kpts, confidences, frame)
        
        return annotated_frame


if __name__ == "__main__":
    # Path default model pose
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Utamakan file engine TensorRT jika ada, jika tidak gunakan .pt
    engine_path = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.engine')
    pt_path = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.pt')
    
    model_path = engine_path if os.path.exists(engine_path) else pt_path
    
    if os.path.exists(model_path):
        pipeline = BiomechanicalPipeline(model_path)
        cap = cv2.VideoCapture(0)
        
        print("🎥 Memulai Real-Time Biomechanical Stream (Tekan 'q' untuk keluar)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            output = pipeline.process_frame(frame)
            cv2.imshow("Biomechanical Analysis Pipeline", output)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"⚠️ Model tidak ditemukan di {model_path}. Harap selesaikan pelatihan terlebih dahulu.")