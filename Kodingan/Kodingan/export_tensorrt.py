import os
import numpy as np
from ultralytics import YOLO

def export_to_tensorrt(dynamic=False, half=True, workspace=4, int8=False):
    """
    Mengonversi model PyTorch (.pt) ke format TensorRT Engine (.engine)
    yang teroptimasi tinggi untuk GPU NVIDIA.
    """
    print("🚀 Memulai proses pengonversian model ke TensorRT Engine...")
    
    # Mendapatkan path absolut root proyek secara dinamis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Path model pemenang (misal: Medium)
    model_path = os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model PyTorch tidak ditemukan di path: {model_path}")
        return None
        
    model = YOLO(model_path)

    # Ekspor ke engine TensorRT
    # Catatan: Wajib menggunakan GPU CUDA (device=0)
    print(f"⚙️ Parameter Ekspor: Dynamic={dynamic}, FP16={half}, INT8={int8}, Workspace={workspace}GB")
    engine_path = model.export(
        format="engine",
        device=0,        # Menggunakan GPU CUDA
        half=half,       # Presisi FP16 untuk kecepatan maksimal
        int8=int8,       # Quantization INT8 jika didukung hardware
        dynamic=dynamic, # Support dynamic input resolution & batch size
        workspace=workspace, # Alokasi VRAM maksimal untuk builder (GB)
        simplify=True    # Menyederhanakan ONNX graph internal
    )
    
    print(f"✅ Konversi berhasil! File engine disimpan di: {engine_path}")
    
    # Verifikasi loading & test inference engine
    if engine_path and os.path.exists(engine_path):
        print("🔍 Memverifikasi integritas file .engine dengan test inference...")
        try:
            trt_model = YOLO(engine_path, task='pose')
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = trt_model(dummy_img, device=0, verbose=False)
            print("✅ Verifikasi Engine Sukses! Engine siap digunakan untuk real-time deployment.")
        except Exception as e:
            print(f"⚠️ Peringatan saat tes inferensi engine: {e}")

    return engine_path

if __name__ == "__main__":
    export_to_tensorrt()