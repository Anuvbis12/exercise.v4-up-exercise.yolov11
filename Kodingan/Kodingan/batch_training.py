import os
from ultralytics import YOLO

def train_yolo_batch():
    """
    Melakukan looping training untuk 5 ukuran model YOLOv11-Pose.
    """
    models = [
        #'yolo11n-pose.pt', 
        #'yolo11s-pose.pt', 
        #'yolo11m-pose.pt', 
        #'yolo11l-pose.pt', 
        'yolo11x-pose.pt'
    ]
    
    # Path ke file data.yaml pada dataset Roboflow
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(base_dir, "exercise.v4-up-exercise.yolov8", "data.yaml")
    epochs = 50 # Diatur ke 50 epoch
    imgsz = 640
    device = 0 # 0
    optimizer = "AdamW"
    batch_size = 8 # Disesuaikan ke 8 untuk stabilitas VRAM model XL (8GB VRAM)
    workers = 0 # 0 agar tidak memakan RAM sistem untuk subprocess
    amp = True # Automatic Mixed Precision (FP16) di GPU
    cache = False # Tanpa RAM cache
    
    # Hiperparameter Optimasi Pose & Loss
    lr0 = 0.002 # Learning rate awal yang disarankan untuk AdamW
    lrf = 0.01 # Learning rate akhir ratio
    cos_lr = True # Cosine learning rate decay (penurunan LR lebih mulus)
    patience = 30 # Early stopping jika metric plateau selama 30 epoch
    close_mosaic = 15 # Nonaktifkan mosaic 15 epoch terakhir agar model fokus pada objek asli
    pose_loss_weight = 12.0 # Bobot Loss Keypoint Pose
    
    # Augmentasi Data khusus Pose Biomekanika
    degrees = 10.0 # Rotasi sudut 10 derajat
    translate = 0.1 # Pergeseran translasi 10%
    scale = 0.5 # Skala zoom 50%
    fliplr = 0.5 # Flip horizontal (keypoints disesuaikan otomatis)

    output_base_dir = "models/trained_weights"
    os.makedirs(output_base_dir, exist_ok=True)

    # Cek ketersediaan CUDA GPU
    import torch
    import gc
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ Menjalankan pada GPU Only: {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("⚠️ GPU CUDA tidak terdeteksi! Training akan berjalan di CPU.")

    for model_name in models:
        print(f"=== Memulai Pelatihan Optimal: {model_name} ===")
        model = YOLO(model_name)
        
        # Eksekusi training teroptimasi
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            batch=batch_size,
            workers=workers,
            amp=amp,
            cache=cache,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            cos_lr=cos_lr,
            patience=patience,
            close_mosaic=close_mosaic,
            pose=pose_loss_weight,
            degrees=degrees,
            translate=translate,
            scale=scale,
            fliplr=fliplr,
            project=output_base_dir,
            name=model_name.split('.')[0]
        )
        print(f"=== Pelatihan {model_name} Selesai ===\n")
        
        # Bersihkan sisa memori VRAM & RAM setelah setiap model selesai
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train_yolo_batch()