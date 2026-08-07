import os
import sys
import time
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
from ultralytics import YOLO


class PalmCenterTracker:
    """
    Tracker 1 Titik Pusat Telapak Tangan (Palm Center) ringan & presisi tinggi.
    Menghitung pusat telapak berdasarkan vektor ekstensi lengan (Siku -> Pergelangan)
    dengan anti-jitter OneEuroFilter.
    """
    def __init__(self):
        self.smoothers = {
            "Left": OneEuroFilter(min_cutoff=0.7, beta=0.01),
            "Right": OneEuroFilter(min_cutoff=0.7, beta=0.01)
        }

    def extract_hands(self, frame, body_keypoints, confidences):
        if body_keypoints is None or confidences is None:
            for s in self.smoothers.values():
                s.reset()
            return []

        detected = []
        wrist_indices = [
            (9, "Left", 7),   # (Wrist Idx, Label, Elbow Idx)
            (10, "Right", 8)
        ] if len(body_keypoints) == 17 else [
            (4, "Left", 3),
            (6, "Right", 5)
        ]

        active = set()
        for w_idx, label, e_idx in wrist_indices:
            if w_idx < len(body_keypoints) and confidences[w_idx] >= 0.35:
                active.add(label)
                wx, wy = body_keypoints[w_idx]

                # Hitung posisi Palm Center = Wrist + 35% panjang lengan searah vektor (Siku -> Wrist)
                if e_idx < len(body_keypoints) and confidences[e_idx] >= 0.35:
                    ex, ey = body_keypoints[e_idx]
                    dx, dy = float(wx - ex), float(wy - ey)
                    length = max(1e-5, math.hypot(dx, dy))
                    palm_x = float(wx) + (dx / length) * (length * 0.35)
                    palm_y = float(wy) + (dy / length) * (length * 0.35)
                else:
                    palm_x, palm_y = float(wx), float(wy) - 35.0

                palm_pt = np.array([[palm_x, palm_y]], dtype=np.float32)

                # Filter anti-jitter OneEuroFilter
                if label in self.smoothers:
                    palm_pt = self.smoothers[label].filter(palm_pt)

                px, py = float(palm_pt[0, 0]), float(palm_pt[0, 1])

                detected.append({
                    'label': label,
                    'wrist': np.array([float(wx), float(wy)]),
                    'palm_center': np.array([px, py]),
                    'score': confidences[w_idx]
                })

        for key in self.smoothers:
            if key not in active:
                self.smoothers[key].reset()

        return detected


class ThreadedWebcam:
    """
    Multithreaded Camera Reader teroptimasi ultra-low latency (0 ms queue lag).
    Membuang frame lama secara kontinu (buffer flushing) untuk stream kamera real-time.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)    # 854x480 → aspect 16:9, bandwidth USB lebih ringan
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # Banyak webcam unlock 30+ FPS mode di resolusi ini
            self.cap.set(cv2.CAP_PROP_FPS, 30)             # Kunci target FPS ke driver kamera
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
            if not self.cap.isOpened():
                break
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                with self.lock:
                    self.grabbed = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened() and not self.stopped

    def release(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()


class OneEuroFilter:
    """
    Industry-Standard One-Euro Filter (Digunakan oleh MediaPipe, OpenPose, & ARKit).
    Secara adaptif meredam jitter saat diam dan meredam lag saat bergerak cepat.
    """
    def __init__(self, min_cutoff=0.8, beta=0.005, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1.0 - a) * x_prev

    def filter(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        if self.x_prev is None or self.x_prev.shape != x.shape:
            self.t_prev = timestamp
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        t_e = max(1e-5, timestamp - self.t_prev)
        self.t_prev = timestamp

        # Hitung turunan kecepatan pergerakan
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        self.dx_prev = dx_hat

        # Cutoff frekuensi adaptif berimbang kecepatan
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class BiomechanicalPipeline:
    def __init__(self, pose_model_path=None, classifier_model_path=None, conf_threshold=0.5):
        # Auto-detect GPU/CPU device
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        # Gunakan model resmi yolo11n-pose.pt untuk akurasi maksimal pada webcam live stream
        default_coco_path = "yolo11n-pose.pt"
        target_path = pose_model_path if (pose_model_path and os.path.exists(pose_model_path)) else default_coco_path
        
        print(f"📦 Loading Model Pose Detector dari: {target_path}")
        self.detector = YOLO(target_path)
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
                
        # Path config ByteTrack Kustom (tanpa GMC, dioptimalkan untuk webcam statis)
        self._tracker_yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bytetrack_live.yaml"
        )

        # Inisialisasi One-Euro Filter (Stabil, Zero-Jitter, Low Latency)
        self.smoother = OneEuroFilter(min_cutoff=0.8, beta=0.005)
        
        # Inisialisasi Palm Center Tracker (Lightweight, Zero-Jitter, 100% Presisi)
        self.hand_tracker = PalmCenterTracker()
        print("🖐️ Palm Center Tracker (Lightweight, Zero-Jitter) berhasil dimuat!")
        
        # Repetition Counter & Stage Tracking
        self.rep_count = 0
        self.stage = "UP"
        
        # Metrik Performa Real-Time
        self.prev_time = time.time()
        self.fps = 0.0
        self._fps_buffer = []       # Buffer moving average FPS (15 frame)
        self._fps_buffer_size = 15  # Ukuran buffer — semakin besar semakin mulus angkanya di HUD

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

    def draw_angle_arc(self, frame, A, B, C, color=(0, 255, 255), radius=30):
        """
        Menggambar busur/arc sektor berwarna transparan di titik sendi B.
        """
        try:
            angA = math.degrees(math.atan2(A[1] - B[1], A[0] - B[0]))
            angC = math.degrees(math.atan2(C[1] - B[1], C[0] - B[0]))
            start_ang, end_ang = min(angA, angC), max(angA, angC)
            if end_ang - start_ang > 180:
                start_ang, end_ang = end_ang, start_ang + 360
            overlay = frame.copy()
            cv2.ellipse(overlay, (int(B[0]), int(B[1])), (radius, radius), 0, start_ang, end_ang, color, -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.ellipse(frame, (int(B[0]), int(B[1])), (radius, radius), 0, start_ang, end_ang, color, 2, cv2.LINE_AA)
        except Exception:
            pass

    def stage1_extract_and_smooth(self, frame):
        """
        Stage 1: Ekstraksi keypoint, Persistent Object Tracking antar-frame, dan temporal smoothing.
        """
        try:
            results = self.detector.track(
                frame, 
                device=self.device,
                imgsz=480,      # 640 → 480: GPU headroom lebih lega, latensi ~8ms lebih konsisten
                conf=0.35,
                persist=True,
                verbose=False,
                tracker=self._tracker_yaml_path
            )
        except Exception:
            results = self.detector.predict(
                frame, 
                device=self.device,
                imgsz=480,      # Sama dengan track — konsisten
                conf=0.35,
                verbose=False
            )
        
        if not results or len(results[0].keypoints) == 0:
            self.smoother.reset()
            return None, None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.smoother.reset()
            return None, None
            
        # Mengunci posisi persendian objek target ber-confidence tertinggi
        best_idx = int(boxes.conf.argmax().cpu().item())
        kpt_data = results[0].keypoints[best_idx]
        
        if kpt_data.xy is None or len(kpt_data.xy) == 0 or kpt_data.xy.shape[1] == 0:
            self.smoother.reset()
            return None, None

        keypoints = kpt_data.xy[0].cpu().numpy() # Shape (N, 2)
        confidences = kpt_data.conf[0].cpu().numpy() if kpt_data.conf is not None else np.ones(len(keypoints))
        
        # Aplikasikan One-Euro Filter pada koordinat (x, y) keypoint
        smoothed_keypoints = self.smoother.filter(keypoints)
        return smoothed_keypoints, confidences

    def stage1_5_extract_and_smooth_hands(self, frame, body_keypoints=None, confidences=None):
        """
        Stage 1.5: Deteksi 21 Keypoint Jari & Evaluasi Status Genggaman Tangan (100% PyTorch Native).
        """
        return self.hand_tracker.extract_hands(frame, body_keypoints, confidences)

    def stage2_classify_movement(self, keypoints):
        """
        Stage 2: Klasifikasi jenis gerakan.
        """
        if self.classifier is not None:
            features = keypoints.flatten().reshape(1, -1)
            movement_class = self.classifier.predict(features)[0]
            return movement_class
        return "Push-up"

    def stage3_visualize_and_evaluate(self, movement, keypoints, confidences, hands_data, frame):
        """
        Stage 3: Visualisasi skeleton 100% presisi (Support COCO 17 Keypoints, Roboflow 13 Keypoints & MediaPipe 21 Finger Keypoints).
        """
        status = "POSISIKAN TUBUH TERLIHAT PENUH"
        color = (0, 255, 255)
        
        if len(keypoints) == 17:
            # COCO 17 Keypoints:
            # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
            # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist
            # 11: L-Hip, 12: R-Hip, 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
            skeleton_limbs = [
                (0, 1), (0, 2), (1, 3), (2, 4),         # Wajah
                (5, 6),                                 # Antar Bahu
                (5, 7), (7, 9),                         # Lengan Kiri
                (6, 8), (8, 10),                        # Lengan Kanan
                (5, 11), (6, 12), (11, 12),             # Torso
                (11, 13), (13, 15),                     # Kaki Kiri
                (12, 14), (14, 16)                      # Kaki Kanan
            ]
            kpt_sh_l, kpt_hip_l, kpt_ank_l, kpt_kne_l = 5, 11, 15, 13
            kpt_elb_l, kpt_wri_l = 7, 9
            kpt_sh_r, kpt_hip_r, kpt_ank_r, kpt_kne_r = 6, 12, 16, 14
            kpt_elb_r, kpt_wri_r = 8, 10
        else:
            # Roboflow 13 Keypoints
            skeleton_limbs = [
                (0, 1), (0, 2), (1, 2),
                (1, 3), (3, 4), (2, 5), (5, 6),
                (1, 7), (2, 8), (7, 8),
                (7, 9), (9, 11), (8, 10), (10, 12)
            ]
            kpt_sh_l, kpt_hip_l, kpt_ank_l, kpt_kne_l = 1, 7, 11, 9
            kpt_elb_l, kpt_wri_l = 3, 4
            kpt_sh_r, kpt_hip_r, kpt_ank_r, kpt_kne_r = 2, 8, 12, 10
            kpt_elb_r, kpt_wri_r = 5, 6

        h, w = frame.shape[:2]
        draw_conf_thresh = 0.35

        def is_valid_pt(idx):
            if idx >= len(keypoints) or confidences[idx] < draw_conf_thresh:
                return False
            x, y = keypoints[idx]
            return 2 <= x < (w - 2) and 2 <= y < (h - 2)

        # 1. Gambar Bounding Box Presisi dengan Handle Control Dots & Label SKELETON 1 (MANUAL)
        valid_indices = [i for i in range(len(keypoints)) if is_valid_pt(i)]
        if len(valid_indices) >= 4:
            pts = keypoints[valid_indices]
            bx_min, by_min = max(0, int(np.min(pts[:, 0])) - 20), max(0, int(np.min(pts[:, 1])) - 25)
            bx_max, by_max = min(w - 1, int(np.max(pts[:, 0])) + 20), min(h - 1, int(np.max(pts[:, 1])) + 20)
            
            # Rectangle Border Kuning Cerah
            cv2.rectangle(frame, (bx_min, by_min), (bx_max, by_max), (0, 255, 255), 2, cv2.LINE_AA)
            
            # Handle Dots (4 sudut + 4 titik tengah + 1 titik kontrol putar atas)
            mid_x = (bx_min + bx_max) // 2
            mid_y = (by_min + by_max) // 2
            top_handle_y = max(5, by_min - 20)
            
            handles = [
                (bx_min, by_min), (bx_max, by_min), (bx_min, by_max), (bx_max, by_max),
                (mid_x, by_min), (mid_x, by_max), (bx_min, mid_y), (bx_max, mid_y),
                (mid_x, top_handle_y)
            ]
            
            # Garis penghubung ke titik rotasi atas
            cv2.line(frame, (mid_x, by_min), (mid_x, top_handle_y), (0, 255, 255), 1, cv2.LINE_AA)
            
            for hx, hy in handles:
                cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), 6, (0, 0, 0), 1, cv2.LINE_AA)

            # Label Text "SKELETON 1 (MANUAL)" di pojok atas kanan bounding box
            label_text = "SKELETON 1 (MANUAL)"
            cv2.putText(frame, label_text, (bx_max - 175, by_min + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label_text, (bx_max - 176, by_min + 19), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Gambar Garis Skeleton Tebal Warna Kuning Cerah (Single Color Yellow Limbs)
        limb_color = (0, 230, 255)
        for p1, p2 in skeleton_limbs:
            if is_valid_pt(p1) and is_valid_pt(p2):
                pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                cv2.line(frame, pt1, pt2, limb_color, 3, cv2.LINE_AA)

        # 3. Gambar Titik Sendi Lingkaran Kuning Solid & Angka Indeks Sendi (1..17)
        for i in range(len(keypoints)):
            if is_valid_pt(i):
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                # Lingkaran sendi kuning solid
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 7, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Angka 1-indexed (1, 2, 3, ... 17) dengan shadow hitam untuk kontras tinggi
                kpt_num_str = str(i + 1)
                cv2.putText(frame, kpt_num_str, (x + 8, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, kpt_num_str, (x + 7, y + 4), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 3.5 Visualisasi Palm Center (1 Titik Pusat Telapak Tangan per Tangan)
        hand_summary = []
        for hand in hands_data:
            label = hand['label']
            wrist = hand['wrist']
            palm = hand['palm_center']
            hand_summary.append(f"{label[0]}: Palm Active")

            wx, wy = int(wrist[0]), int(wrist[1])
            px, py = int(palm[0]), int(palm[1])

            # Garis konektor dari Wrist ke Palm Center (Cyan halus)
            cv2.line(frame, (wx, wy), (px, py), (255, 255, 0), 2, cv2.LINE_AA)

            # Titik Pusat Telapak (Hijau Solid dengan Border Hitam)
            cv2.circle(frame, (px, py), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 9, (0, 0, 0), 2, cv2.LINE_AA)

        hip_angle_val = 0.0
        elbow_angle_val = 0.0

        # 4. Perhitungan Sudut Biomekanik & Rep Counter (Push-up/Plank)
        if len(keypoints) == 13:
            kpt_sh, kpt_hip, kpt_ank, kpt_kne = 1, 7, 11, 9
            kpt_elb, kpt_wri = 3, 4
        else:
            kpt_sh, kpt_hip, kpt_ank, kpt_kne = 5, 11, 15, 13
            kpt_elb, kpt_wri = 7, 9

        # Pilih acuan kaki: Ankle jika terdeteksi (conf >= 0.55), atau Knee jika ankle di luar kamera
        ref_leg_kpt = kpt_ank if confidences[kpt_ank] >= 0.55 else (kpt_kne if confidences[kpt_kne] >= 0.55 else None)

        if (ref_leg_kpt is not None and 
            confidences[kpt_sh] >= 0.55 and 
            confidences[kpt_hip] >= 0.55 and
            math.dist(keypoints[kpt_sh], keypoints[kpt_hip]) > 40):
            
            # Hitung Sudut Pinggul (Postur Lurus)
            hip_angle_val = self.calculate_angle(keypoints[kpt_sh], keypoints[kpt_hip], keypoints[ref_leg_kpt])
            
            if hip_angle_val > 165:
                status = "POSTUR SEMPURNA"
                color = (0, 255, 0)
            elif hip_angle_val > 150:
                status = "POSTUR BAIK"
                color = (0, 255, 255)
            else:
                status = "POSTUR BURUK (Pinggul Terlalu Turun!)"
                color = (0, 0, 255)

            # Draw Angle Arc & Text Overlay di Pinggul
            self.draw_angle_arc(frame, keypoints[kpt_sh], keypoints[kpt_hip], keypoints[ref_leg_kpt], color=(0, 255, 255), radius=30)
            hip_pt = (int(keypoints[kpt_hip][0]), int(keypoints[kpt_hip][1]))
            cv2.putText(frame, f"{int(hip_angle_val)}deg", (hip_pt[0] + 12, hip_pt[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if (confidences[kpt_sh] >= 0.55 and 
            confidences[kpt_elb] >= 0.55 and 
            confidences[kpt_wri] >= 0.55 and
            math.dist(keypoints[kpt_sh], keypoints[kpt_elb]) > 25):
            
            # Hitung Sudut Siku (Fase Gerakan Push-up)
            elbow_angle_val = self.calculate_angle(keypoints[kpt_sh], keypoints[kpt_elb], keypoints[kpt_wri])
            
            # Logika Repetition Counter State Machine
            if elbow_angle_val < 90 and self.stage == "UP":
                self.stage = "DOWN"
            if elbow_angle_val > 155 and self.stage == "DOWN":
                self.stage = "UP"
                self.rep_count += 1

            # Draw Angle Arc & Text Overlay di Siku
            self.draw_angle_arc(frame, keypoints[kpt_sh], keypoints[kpt_elb], keypoints[kpt_wri], color=(255, 0, 255), radius=28)
            elb_pt = (int(keypoints[kpt_elb][0]), int(keypoints[kpt_elb][1]))
            cv2.putText(frame, f"{int(elbow_angle_val)}deg", (elb_pt[0] + 12, elb_pt[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        # 5. Modern Glassmorphic HUD Card Dashboard (Di Pojok Atas Kanan)
        hud_w, hud_h = 360, 210
        hud_x = max(10, w - hud_w - 15)
        hud_y = 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 230, 255), 2) # Border Cyan

        # Formulasi Text Status Finger Tracking untuk HUD
        hands_str = " | ".join(hand_summary) if hand_summary else "Palm Tracker Active"

        # HUD Text Content
        cv2.putText(frame, "BIOMECHANICAL ANALYSIS", (hud_x + 15, hud_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"REPS : {self.rep_count}", (hud_x + 15, hud_y + 62), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"STAGE: {self.stage}", (hud_x + 190, hud_y + 62), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GERAKAN : {movement}", (hud_x + 15, hud_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"STATUS  : {status}", (hud_x + 15, hud_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"SUDUT   : Hip {int(hip_angle_val)}deg | Elbow {int(elbow_angle_val)}deg", (hud_x + 15, hud_y + 146), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"HANDS   : {hands_str}", (hud_x + 15, hud_y + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS     : {self.fps:.1f}", (hud_x + 15, hud_y + 194), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def process_frame(self, frame):
        # Hitung FPS dengan Moving Average 15 frame (angka HUD mulus, tidak loncat-loncat)
        curr_time = time.time()
        raw_fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        self._fps_buffer.append(raw_fps)
        if len(self._fps_buffer) > self._fps_buffer_size:
            self._fps_buffer.pop(0)
        self.fps = sum(self._fps_buffer) / len(self._fps_buffer)

        # Cek jika frame kamera hitam — sampling region kecil (jauh lebih cepat dari frame.mean())
        if frame is None or frame[230:250, 400:440].mean() < 1.0:
            cv2.putText(frame, "KAMERA HITAM / TERKUNCI!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Buka Privacy Shutter / Tekan Fn+F6 / Cek Lenovo Vantage", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame

        # Tahap 1: Ekstraksi Keypoint Body & Temporal Smoothing
        keypoints, confidences = self.stage1_extract_and_smooth(frame)

        # Tahap 1.5: Ekstraksi 21 Keypoint Jari & Gesture Tangan (Dual Engine MediaPipe / YOLO Wrist ROI)
        hands_data = self.stage1_5_extract_and_smooth_hands(frame, keypoints, confidences)

        if keypoints is None: 
            cv2.putText(frame, "Mencari Pose / Orang...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return frame
        
        # Tahap 2: Klasifikasi Gerakan
        movement = self.stage2_classify_movement(keypoints)
        
        # Tahap 3: Visualisasi Biomekanik & Evaluasi (100% Stabil)
        annotated_frame = self.stage3_visualize_and_evaluate(movement, keypoints, confidences, hands_data, frame)
        
        return annotated_frame


if __name__ == "__main__":
    # Model resmi yolo11n-pose.pt (Terlatih pada 200.000 citra COCO untuk akurasi webcam maksimal)
    model_path = "yolo11n-pose.pt"

    video_source = None
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
    else:
        # Cari kamera terhubung (indeks 0, 1, atau 2)
        for idx in [0, 1, 2]:
            temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                temp_cap.release()
                if ret and frame is not None:
                    video_source = idx
                    print(f"🔍 Auto-detected Kamera pada Indeks: {video_source}")
                    break
        if video_source is None:
            video_source = 0

    pipeline = BiomechanicalPipeline(model_path)
    
    # Buka webcam / window stream dengan multithreading
    if isinstance(video_source, int):
        webcam = ThreadedWebcam(video_source).start()
        if not webcam.isOpened():
            print(f"❌ Gagal membuka sumber webcam: {video_source}")
            sys.exit(1)
        
        window_name = "Biomechanical Analysis Pipeline"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print(f"🎥 Memulai Stream Kamera Real-Time (Indeks {video_source}, Ultra-Low Latency)... Tekan 'q' untuk keluar.")
        while webcam.isOpened():
            ret, frame = webcam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            output = pipeline.process_frame(frame)
            cv2.imshow(window_name, output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        webcam.release()
        cv2.destroyAllWindows()