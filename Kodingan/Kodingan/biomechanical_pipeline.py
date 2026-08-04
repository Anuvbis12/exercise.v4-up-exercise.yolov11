import os
import sys
import time
import cv2
import numpy as np
import torch
import math
import threading
from ultralytics import YOLO

class ThreadedWebcam:
    """
    Multithreaded Camera Reader untuk menghilangkan blocking lag cap.read() pada Windows.
    Frame ditangkap di background thread secara asinkron (0 ms latency di main loop).
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.grabbed, self.frame = self.cap.read()
        else:
            self.grabbed, self.frame = False, None
            
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        if self.cap.isOpened():
            t = threading.Thread(target=self.update, daemon=True)
            t.start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            with self.lock:
                self.grabbed = ret
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            ret = self.grabbed
            frame = self.frame.copy() if self.frame is not None else None
        return ret, frame

    def release(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
            
    def isOpened(self):
        return self.cap.isOpened()

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
        Stage 1: Ekstraksi keypoint dan temporal smoothing teroptimasi.
        """
        # Eksekusi inferensi teroptimasi (imgsz=320 untuk kecepatan super tinggi)
        results = self.detector.predict(
            frame, 
            device=self.device,
            imgsz=320,
            conf=self.conf_threshold,
            verbose=False
        )
        
        if not results or len(results[0].keypoints) == 0:
            self.smoother.reset()
            return None, None
            
        kpt_data = results[0].keypoints[0] # Ambil deteksi pose orang pertama
        keypoints = kpt_data.xy[0].cpu().numpy() # Shape (N, 2)
        confidences = kpt_data.conf[0].cpu().numpy() if kpt_data.conf is not None else np.ones(len(keypoints))
        
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
        # Indeks acuan: Bahu Kiri/Kanan, Pinggul Kiri/Kanan
        ref_indices = [1, 2, 7, 8] if len(keypoints) == 13 else [5, 6, 11, 12]
        
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
        Stage 3b: Evaluasi postur berdasarkan JAA (Joint Angle Accuracy) dan Visualisasi Skeleton.
        """
        status = "TERDETEKSI"
        color = (0, 255, 255)
        
        # Pasangan koneksi tulang skeleton
        skeleton_limbs = [
            (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), # Lengan atas/bawah
            (1, 7), (2, 8), (7, 8),                # Torso / Bahu ke Pinggul
            (7, 9), (9, 11), (8, 10), (10, 12)     # Kaki / Paha & Betis
        ] if len(keypoints) == 13 else [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        # Gambar Garis Skeleton
        for p1, p2 in skeleton_limbs:
            if p1 < len(keypoints) and p2 < len(keypoints):
                if confidences[p1] > self.conf_threshold and confidences[p2] > self.conf_threshold:
                    pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                    pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Gambar Titik Sendi (Keypoint Circles)
        for i, (x, y) in enumerate(keypoints):
            if confidences[i] > self.conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)

        if movement == "Push-up":
            # Indeks: Bahu, Pinggul, Pergelangan Kaki (Ankle)
            kpt_shoulder, kpt_hip, kpt_ankle = (1, 7, 11) if len(keypoints) == 13 else (5, 11, 15)

            if (confidences[kpt_shoulder] > self.conf_threshold and 
                confidences[kpt_hip] > self.conf_threshold and 
                confidences[kpt_ankle] > self.conf_threshold):
                hip_angle = self.calculate_angle(keypoints[kpt_shoulder], keypoints[kpt_hip], keypoints[kpt_ankle])
                
                if hip_angle > 165:
                    status = "POSTUR SEMPURNA"
                    color = (0, 255, 0)
                elif hip_angle > 150:
                    status = "POSTUR BAIK"
                    color = (0, 255, 255)
                else:
                    status = "POSTUR BURUK (Pinggul Terlalu Turun!)"
                    color = (0, 0, 255)

        # Visualisasi Overlay Real-Time
        cv2.putText(frame, f"Gerakan: {movement}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def process_frame(self, frame):
        # Hitung FPS
        curr_time = time.time()
        self.fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time

        # Cek jika frame kamera hitam (shutter tertutup / privacy mode active)
        # Jangan jalankan inferensi YOLO jika kamera hitam untuk menghemat CPU/GPU & hindari lag!
        if frame is None or frame.mean() < 1.0:
            cv2.putText(frame, "KAMERA HITAM / TERKUNCI!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Buka Privacy Shutter / Tekan Fn+F6 / Cek Lenovo Vantage", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame

        # Tahap 1: Ekstraksi & EMA Smoothing (Hanya jika frame valid)
        keypoints, confidences = self.stage1_extract_and_smooth(frame)

        if keypoints is None: 
            # Tampilkan pesan jika belum/tidak ada pose terdeteksi
            cv2.putText(frame, "Mencari Pose / Orang...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
    
    # Pilih model: Utamakan model Nano (yolo11n-pose-5) yang sangat ringan & cepat (high FPS)
    nano_pt = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11n-pose-5', 'weights', 'best.pt')
    medium_pt = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.pt')
    engine_path = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.engine')
    
    if os.path.exists(nano_pt):
        model_path = nano_pt
    elif os.path.exists(medium_pt):
        model_path = medium_pt
    else:
        model_path = engine_path
    
    # Sumber Video: Argumen arg1 jika ada (contoh: python biomechanical_pipeline.py video.mp4), jika tidak webcam 0
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    if os.path.exists(model_path):
        pipeline = BiomechanicalPipeline(model_path)
        
        # Buka video atau webcam dengan multithreading
        if isinstance(video_source, int):
            webcam = ThreadedWebcam(video_source).start()
            if not webcam.isOpened():
                print(f"❌ Gagal membuka sumber webcam: {video_source}")
                sys.exit(1)
            
            print(f"🎥 Memulai Real-Time Webcam Stream (Multithreaded 60 FPS)... Tekan 'q' untuk keluar.")
            while webcam.isOpened():
                ret, frame = webcam.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                    
                output = pipeline.process_frame(frame)
                cv2.imshow("Biomechanical Analysis Pipeline", output)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
            webcam.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"❌ Gagal membuka sumber video: {video_source}")
                sys.exit(1)

            print(f"🎥 Memulai Stream File Video [{video_source}]... Tekan 'q' untuk keluar.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None: break
                    
                output = pipeline.process_frame(frame)
                cv2.imshow("Biomechanical Analysis Pipeline", output)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
            cap.release()
            cv2.destroyAllWindows()
    else:
        print(f"⚠️ Model tidak ditemukan di {model_path}. Harap selesaikan pelatihan terlebih dahulu.")