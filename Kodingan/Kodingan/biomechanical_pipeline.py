import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
import math

class BiomechanicalPipeline:
    def __init__(self, pose_model_path, classifier_model_path):
        # Load Model Deteksi YOLOv11-Pose
        self.detector = YOLO(pose_model_path)
        
        # Load Dummy/Pre-trained Classifier (Misal: RandomForest)
        # Seharusnya di-load menggunakan pickle/joblib di lingkungan produksi
        self.classifier = RandomForestClassifier() 
        
        # Inisialisasi history untuk Temporal Smoothing
        self.keypoint_history = []
        
    def stage1_extract_and_smooth(self, frame):
        """
        Stage 1: Ekstraksi 17 keypoint COCO dan temporal smoothing via GPU.
        """
        # Eksekusi inferensi murni di GPU dengan FP16
        results = self.detector.predict(
            frame, 
            device=0,       # Paksa menggunakan GPU 0
            half=True,      # Gunakan FP16 (AMP) untuk kecepatan inferensi
            verbose=False
        )
        
        if not results or results[0].keypoints is None:
            return None
            
        # Pindahkan tensor dari GPU ke memori RAM (CPU) hanya saat akan diolah oleh Numpy
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        self.keypoint_history.append(keypoints)
        if len(self.keypoint_history) > 5:
            self.keypoint_history.pop(0)
            
        smoothed_keypoints = np.mean(self.keypoint_history, axis=0)
        return smoothed_keypoints

    def stage2_classify_movement(self, keypoints):
        """
        Stage 2: Klasifikasi jenis gerakan secara sekuensial.
        Gatekeeper ini menentukan logika Stage 3.
        """
        # Flatten keypoints untuk input classifier
        features = keypoints.flatten().reshape(1, -1)
        # Dummy prediction untuk struktur - ganti dengan inference model asli
        # movement_types = ["Bird Dog", "Knee Push-up", "Plank", "Push-up", "Reverse Lunge"]
        movement_class = "Push-up" 
        return movement_class

    def stage3_planar_homography(self, keypoints):
        """
        Stage 3a: Mengoreksi distorsi kamera miring menggunakan DLT & RANSAC.
        Memetakan dari ruang citra miring ke tampilan samping (canonical side-view).
        """
        # Mendefinisikan titik sumber (dari gambar) dan titik tujuan (bidang datar)
        # Implementasi ini membutuhkan titik referensi lingkungan yang valid
        # Sebagai contoh arsitektur, kita aplikasikan matriks pseudo-homografi RANSAC
        src_pts = np.array([keypoints[5], keypoints[6], keypoints[11], keypoints[12]])
        dst_pts = np.array([[0, 0], [100, 0], [0, 200], [100, 200]])
        
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            # Transformasikan semua keypoint menggunakan Pseudo-Homografi H
            kpts_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
            transformed = (H @ kpts_homogeneous.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]
            return transformed
        return keypoints

    def calculate_angle(self, A, B, C):
        """
        Menghitung sudut menggunakan Hukum Kosinus 2D.
        A, B, C adalah array koordinat [x, y]. B adalah titik sendi.
        """
        a = math.dist(B, C)
        c = math.dist(A, B)
        b = math.dist(A, C)
        
        if a == 0 or c == 0:
            return 0
            
        cos_val = (a**2 + c**2 - b**2) / (2 * a * c)
        angle = math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0)))
        return angle

    def stage3_biomechanical_jaa(self, movement, keypoints, frame):
        """
        Stage 3b: Evaluasi postur berdasarkan JAA (Joint Angle Accuracy).
        Hanya menghitung sudut yang relevan untuk gerakan spesifik.
        """
        status = "TIDAK DIKETAHUI"
        color = (0, 0, 255)
        
        if movement == "Push-up":
            # Evaluasi siku: Bahu (5) - Siku (7) - Pergelangan Tangan (9)
            # Evaluasi pinggul (Trunk): Bahu (5) - Pinggul (11) - Pergelangan Kaki (15)
            elbow_angle = self.calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            hip_angle = self.calculate_angle(keypoints[5], keypoints[11], keypoints[15])
            
            if hip_angle > 165:
                status = "SEMPURNA"
                color = (0, 255, 0)
            elif hip_angle > 150:
                status = "BAIK"
                color = (0, 255, 255)
            else:
                status = "BURUK (Pinggul terlalu turun!)"
                
        # (Tambahkan rules untuk Plank, Bird Dog, dll sesuai dokumen di sini)

        # Visualisasi Overlay Real-Time
        cv2.putText(frame, f"Gerakan: {movement}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame

    def process_frame(self, frame):
        # Tahap 1
        keypoints = self.stage1_extract_and_smooth(frame)
        if keypoints is None: return frame
        
        # Tahap 2
        movement = self.stage2_classify_movement(keypoints)
        
        # Tahap 3
        normalized_kpts = self.stage3_planar_homography(keypoints)
        annotated_frame = self.stage3_biomechanical_jaa(movement, normalized_kpts, frame)
        
        return annotated_frame

if __name__ == "__main__":
    pipeline = BiomechanicalPipeline("models/trained_weights/yolo11n-pose/weights/best.pt", "classifier.pkl")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.get()
        if not ret: break
            
        output = pipeline.process_frame(frame)
        cv2.imshow("Biomechanical Analysis", output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()