import os
from ultralytics import YOLO

def export_to_tensorrt():
    print("Mulai mengonversi model ke TensorRT...")
    
    # Mendapatkan path absolut root proyek secara dinamis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Path model pemenang (misal: Medium)
    model_path = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model tidak ditemukan di path: {model_path}")
        return
        
    model = YOLO(model_path)

    # Ekspor ke engine
    # Catatan: Ekspor ke format TensorRT (.engine) Wajib menggunakan GPU CUDA (device=0)
    model.export(
        format="engine",
        device=0,        # Pastikan menggunakan GPU
        half=True,       # Presisi FP16 (Wajib untuk kecepatan maksimal)
        workspace=4,     # Alokasi VRAM maksimal untuk konversi (4GB cukup aman)
        simplify=True    # Menyederhanakan arsitektur model
    )
    print("Konversi selesai! File .engine siap digunakan.")

if __name__ == "__main__":
    export_to_tensorrt()