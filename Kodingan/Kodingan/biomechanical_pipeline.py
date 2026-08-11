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


# ── Warna skeleton & HUD per orang (BGR) ─────────────────────────────────────────────
_PERSON_COLORS = [
    (0, 230, 255),    # Person 0 — Cyan Kuning (orisinal)
    (80, 255, 80),    # Person 1 — Hijau Neon
    (0, 140, 255),    # Person 2 — Orange
]


class PersonState:
    """
    Menyimpan seluruh state analisis biomekanik untuk 1 orang tertentu.
    Di-instansiasi per track_id; dihapus otomatis setelah PERSON_TIMEOUT detik.
    """
    def __init__(self):
        self.smoother      = OneEuroFilter(min_cutoff=0.8, beta=0.005)
        self.hand_tracker  = PalmCenterTracker()
        self.rep_count     = 0
        self.stage         = "UP"
        self._depth_buffer = []
        self.depth_pct     = 100.0
        self.last_seen     = time.time()  # Deteksi terakhir untuk GC otomatis


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

        # Inisialisasi state multi-person
        self.persons        = {}   # dict: track_id (int) → PersonState
        self.MAX_PERSONS    = 3    # Batas maksimal orang yang diproses bersamaan
        self.PERSON_TIMEOUT = 5.0  # Detik sebelum state orang yang hilang dihapus

        # Konstanta sudut Push-Up Depth Indicator (dibagi semua orang)
        self.PUSHUP_ANGLE_DOWN = 60.0
        self.PUSHUP_ANGLE_UP   = 160.0

        # Metrik Performa Real-Time (FPS counter)
        self.prev_time        = time.time()
        self.fps              = 0.0
        self._fps_buffer      = []   # Moving average FPS (15 frame)
        self._fps_buffer_size = 15

        print("👥 Multi-Person Tracker: maks. 3 orang aktif")

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
        Stage 1: Deteksi pose semua orang (maks. MAX_PERSONS), tracking per ID,
        temporal smoothing per person, dan GC state orang yang sudah hilang.
        Return: list of dict {track_id, keypoints, confidences}
        """
        try:
            results = self.detector.track(
                frame,
                device=self.device,
                imgsz=480,
                conf=0.35,
                persist=True,
                verbose=False,
                tracker=self._tracker_yaml_path
            )
        except Exception:
            results = self.detector.predict(
                frame,
                device=self.device,
                imgsz=480,
                conf=0.35,
                verbose=False
            )

        if not results or len(results[0].keypoints) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # ── Ambil track IDs (ByteTrack) ───────────────────────────────────────────
        confs   = boxes.conf.cpu().numpy()
        if boxes.id is not None:
            tids = boxes.id.int().cpu().numpy()
        else:
            # Fallback saat tracking ID belum tersedia (frame pertama)
            tids = np.arange(len(confs), dtype=int)

        # Urutkan berdasarkan confidence tertinggi, ambil maks. MAX_PERSONS
        order = np.argsort(confs)[::-1][:self.MAX_PERSONS]

        now = time.time()
        detected = []

        for rank, idx in enumerate(order):
            tid = int(tids[idx])
            kpt_data = results[0].keypoints[idx]
            if kpt_data.xy is None or kpt_data.xy.shape[1] == 0:
                continue

            keypoints   = kpt_data.xy[0].cpu().numpy()
            confidences = (kpt_data.conf[0].cpu().numpy()
                           if kpt_data.conf is not None
                           else np.ones(len(keypoints)))

            # Buat PersonState baru jika ID belum dikenal
            if tid not in self.persons:
                self.persons[tid] = PersonState()

            pstate = self.persons[tid]
            pstate.last_seen = now

            # Aplikasikan OneEuroFilter milik orang ini
            smoothed = pstate.smoother.filter(keypoints)

            detected.append({
                'track_id':   tid,
                'rank':       rank,          # 0 = paling percaya diri
                'keypoints':  smoothed,
                'confidences': confidences,
            })

        # ── GC: hapus state orang yang sudah hilang > PERSON_TIMEOUT ────────────
        stale = [tid for tid, ps in self.persons.items()
                 if (now - ps.last_seen) > self.PERSON_TIMEOUT]
        for tid in stale:
            del self.persons[tid]

        return detected

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

    def stage3_visualize_and_evaluate(self, movement, keypoints, confidences, hands_data, frame,
                                       pstate: 'PersonState' = None, person_rank: int = 0):
        """
        Stage 3: Visualisasi skeleton presisi + evaluasi biomekanik per orang.
        pstate: PersonState orang ini. person_rank: 0/1/2 menentukan warna & posisi HUD.
        """
        status = "POSISIKAN TUBUH TERLIHAT PENUH"
        color  = (0, 255, 255)

        # Warna identitas orang ini
        p_color = _PERSON_COLORS[min(person_rank, len(_PERSON_COLORS) - 1)]
        
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
            cv2.line(frame, (mid_x, by_min), (mid_x, top_handle_y), p_color, 1, cv2.LINE_AA)

            for hx, hy in handles:
                cv2.circle(frame, (hx, hy), 5, p_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), 6, (0, 0, 0), 1, cv2.LINE_AA)

            # Label SKELETON dengan nomor orang
            label_text = f"SKELETON {person_rank + 1}"
            cv2.putText(frame, label_text, (bx_max - 145, by_min + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label_text, (bx_max - 146, by_min + 19), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Gambar Garis Skeleton dengan warna identitas orang
        limb_color = p_color
        for p1, p2 in skeleton_limbs:
            if is_valid_pt(p1) and is_valid_pt(p2):
                pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                cv2.line(frame, pt1, pt2, limb_color, 3, cv2.LINE_AA)

        # 3. Gambar Titik Sendi dengan warna identitas orang
        for i in range(len(keypoints)):
            if is_valid_pt(i):
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                cv2.circle(frame, (x, y), 6, p_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 7, (0, 0, 0), 1, cv2.LINE_AA)
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
            kpt_sh,   kpt_hip, kpt_ank, kpt_kne = 1, 7, 11, 9
            kpt_elb,  kpt_wri = 3, 4
            kpt_sh_r, kpt_elb_r, kpt_wri_r = 2, 3, 4  # Fallback simetris untuk 13-kpt
        else:
            kpt_sh,   kpt_hip, kpt_ank, kpt_kne = 5, 11, 15, 13
            kpt_elb,  kpt_wri = 7, 9
            kpt_sh_r, kpt_elb_r, kpt_wri_r = 6, 8, 10  # Siku/wrist kanan (COCO 17-kpt)

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

        # ── Sudut Siku Kiri ──────────────────────────────────────────────────────
        elbow_samples = []  # Kumpulkan sudut dari sisi yang terdeteksi untuk dirata-ratakan

        if (confidences[kpt_sh] >= 0.55 and 
            confidences[kpt_elb] >= 0.55 and 
            confidences[kpt_wri] >= 0.55 and
            math.dist(keypoints[kpt_sh], keypoints[kpt_elb]) > 25):

            angle_left = self.calculate_angle(keypoints[kpt_sh], keypoints[kpt_elb], keypoints[kpt_wri])
            elbow_samples.append(angle_left)

            # Draw Angle Arc & Text Overlay di Siku Kiri
            self.draw_angle_arc(frame, keypoints[kpt_sh], keypoints[kpt_elb], keypoints[kpt_wri], color=(255, 0, 255), radius=28)
            elb_pt = (int(keypoints[kpt_elb][0]), int(keypoints[kpt_elb][1]))
            cv2.putText(frame, f"{int(angle_left)}deg", (elb_pt[0] + 12, elb_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        # ── Sudut Siku Kanan ─────────────────────────────────────────────────────
        if (kpt_elb_r < len(keypoints) and
            confidences[kpt_sh_r] >= 0.55 and
            confidences[kpt_elb_r] >= 0.55 and
            confidences[kpt_wri_r] >= 0.55 and
            math.dist(keypoints[kpt_sh_r], keypoints[kpt_elb_r]) > 25):

            angle_right = self.calculate_angle(keypoints[kpt_sh_r], keypoints[kpt_elb_r], keypoints[kpt_wri_r])
            elbow_samples.append(angle_right)

            # Draw Angle Arc & Text Overlay di Siku Kanan
            self.draw_angle_arc(frame, keypoints[kpt_sh_r], keypoints[kpt_elb_r], keypoints[kpt_wri_r], color=(200, 0, 255), radius=28)
            elb_r_pt = (int(keypoints[kpt_elb_r][0]), int(keypoints[kpt_elb_r][1]))
            cv2.putText(frame, f"{int(angle_right)}deg", (elb_r_pt[0] + 12, elb_r_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 255), 2, cv2.LINE_AA)

        # ── Rata-rata sudut siku & State Machine ─────────────────────────────────
        if elbow_samples:
            elbow_angle_val = sum(elbow_samples) / len(elbow_samples)

            # Logika Repetition Counter State Machine (per-person via pstate)
            if pstate is not None:
                if elbow_angle_val < 90 and pstate.stage == "UP":
                    pstate.stage = "DOWN"
                if elbow_angle_val > 155 and pstate.stage == "DOWN":
                    pstate.stage = "UP"
                    pstate.rep_count += 1
            else:
                # Fallback jika dipanggil tanpa pstate (single-person legacy)
                pass

        # ── Hitung Depth Percentage dari rata-rata sudut siku ────────────────────
        if elbow_angle_val > 0 and pstate is not None:
            raw_depth = (elbow_angle_val - self.PUSHUP_ANGLE_DOWN) / (
                         self.PUSHUP_ANGLE_UP - self.PUSHUP_ANGLE_DOWN) * 100.0
            raw_depth = max(0.0, min(100.0, raw_depth))
            pstate._depth_buffer.append(raw_depth)
            if len(pstate._depth_buffer) > 5:
                pstate._depth_buffer.pop(0)
            pstate.depth_pct = sum(pstate._depth_buffer) / len(pstate._depth_buffer)

        # 5. HUD Card per orang (diperkecil 30%, ditumpuk vertikal)
        #    Ukuran asli 360x210 → 252x147
        hud_w, hud_h = 252, 147
        hud_x = max(10, w - hud_w - 12)
        hud_y = 12 + person_rank * (hud_h + 5)   # Ditumpuk vertikal per orang

        cur_stage  = pstate.stage      if pstate else "UP"
        cur_reps   = pstate.rep_count  if pstate else 0
        cur_depth  = pstate.depth_pct  if pstate else 100.0

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), p_color, 2)  # Border warna identitas

        hands_str = " | ".join(hand_summary) if hand_summary else "Palm Active"

        # HUD Text (font dikecilkan 30%)
        cv2.putText(frame, f"P{person_rank+1} BIOMECHANICAL",
            (hud_x + 8, hud_y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.42, p_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"REPS:{cur_reps}",
            (hud_x + 8, hud_y + 42), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{cur_stage}",
            (hud_x + 130, hud_y + 42), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"MOV : {movement}",
            (hud_x + 8, hud_y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"STAT: {status[:26]}",
            (hud_x + 8, hud_y + 81), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"ELBOW:{int(elbow_angle_val)}d HIP:{int(hip_angle_val)}d",
            (hud_x + 8, hud_y + 99), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"HAND: {hands_str[:22]}",
            (hud_x + 8, hud_y + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS : {self.fps:.1f}",
            (hud_x + 8, hud_y + 133), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1, cv2.LINE_AA)

        # 6. Depth Bar (per orang, side-by-side di pojok kanan bawah)
        #    Person 0: paling kanan, Person 1: geser kiri 40px, Person 2: geser lagi
        bar_w     = 22
        bar_h     = 160
        bar_x     = w - 40 - person_rank * 40   # Jarak 40px antar bar
        bar_y_top = h - bar_h - 50

        # Background glassmorphic
        bg_pad = 8
        ov2 = frame.copy()
        cv2.rectangle(ov2,
            (bar_x - bg_pad, bar_y_top - 28),
            (bar_x + bar_w + bg_pad, bar_y_top + bar_h + 38),
            (15, 15, 20), -1)
        cv2.addWeighted(ov2, 0.70, frame, 0.30, 0, frame)

        # Track (area kosong)
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + bar_w, bar_y_top + bar_h), (55, 55, 55), -1)
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + bar_w, bar_y_top + bar_h), p_color, 2)

        # Warna fill bar berdasarkan depth
        if cur_depth < 30:
            bar_color = (30, 30, 220)
        elif cur_depth < 65:
            bar_color = (0, 185, 255)
        else:
            bar_color = (30, 220, 80)

        # Fill dari bawah ke atas
        fill_h = int(bar_h * cur_depth / 100.0)
        fill_y = bar_y_top + bar_h - fill_h
        if fill_h > 0:
            cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y_top + bar_h), bar_color, -1)

        # Label stage atas
        cv2.putText(frame, "UP" if cur_stage == "UP" else "DN",
            (bar_x + 1, bar_y_top - 8), cv2.FONT_HERSHEY_DUPLEX, 0.38, p_color, 1, cv2.LINE_AA)

        # Label P# di atas stage
        cv2.putText(frame, f"P{person_rank+1}",
            (bar_x + 3, bar_y_top - 20), cv2.FONT_HERSHEY_DUPLEX, 0.35, p_color, 1, cv2.LINE_AA)

        # Label % bawah
        cv2.putText(frame, f"{int(cur_depth)}%",
            (bar_x - 3, bar_y_top + bar_h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, bar_color, 2, cv2.LINE_AA)

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

        # Cek jika frame kamera hitam — sampling region kecil
        if frame is None or frame[230:250, 400:440].mean() < 1.0:
            cv2.putText(frame, "KAMERA HITAM / TERKUNCI!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Buka Privacy Shutter / Tekan Fn+F6 / Cek Lenovo Vantage", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame

        # Tahap 1: Deteksi & tracking semua orang (maks. 3)
        persons_list = self.stage1_extract_and_smooth(frame)

        if not persons_list:
            cv2.putText(frame, "Mencari Pose / Orang...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return frame

        # Tahap 2-3: Proses setiap orang secara berurutan
        for person in persons_list:
            tid   = person['track_id']
            rank  = person['rank']
            kpts  = person['keypoints']
            conf  = person['confidences']
            pstate = self.persons[tid]

            # 1.5: Ekstraksi Palm Center dari hand_tracker milik orang ini
            hands_data = pstate.hand_tracker.extract_hands(frame, kpts, conf)

            # 2: Klasifikasi gerakan
            movement = self.stage2_classify_movement(kpts)

            # 3: Visualisasi & evaluasi (per orang)
            frame = self.stage3_visualize_and_evaluate(
                movement, kpts, conf, hands_data, frame, pstate, rank
            )

        return frame


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