import os
import torch
import gc
from ultralytics import YOLO

def train_yolo_batch():
    """
    Melakukan looping training teroptimasi untuk ukuran model YOLOv11-Pose.
    """
    # Aktifkan cuDNN benchmark untuk optimasi kecepatan konvolusi GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    models = [
        #'yolo11n-pose.pt', 
        #'yolo11s-pose.pt', 
        #'yolo11m-pose.pt', 
        'yolo11l-pose.pt', 
        #'yolo11x-pose.pt'
    ]
    
    # Path ke file data.yaml pada dataset Roboflow
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(base_dir, "exercise.v4-up-exercise.yolov8", "data.yaml")
    epochs = 200 
    imgsz = 640
    device = 0 
    optimizer = "AdamW"
    batch_size = 8 # Disesuaikan ke 8 untuk stabilitas VRAM model Large/XL (8GB VRAM)
    workers = 2 # Dioptimalkan ke 4 workers untuk mencegah I/O bottleneck DataLoader
    amp = True # Automatic Mixed Precision (FP16) di GPU
    cache = False # Set 'ram' atau 'disk' jika RAM mencukupi untuk mempercepat epoch
    save_period = 10 # Simpan checkpoint berkala setiap 10 epoch
    plots = True # Generasi grafik training & matriks konfusi otomatis
    
    # Hiperparameter Regularisasi & Optimasi Pose (Target mAP 90-93% Robust)
    lr0 = 0.0010 # LR awal diperhalus untuk stabilitas warmup
    lrf = 0.01 # Learning rate akhir ratio
    warmup_epochs = 6.0 # Perpanjang warmup agar kurva loss di awal mulus
    cos_lr = True # Cosine learning rate decay
    patience = 30 # Early stopping jika metric plateau selama 30 epoch
    weight_decay = 0.002 # Regularisasi L2 lebih tinggi untuk menekan over-fitting
    dropout = 0.15 # Dropout 15% pada head model untuk mencegah hafalan piksel
    pose_loss_weight = 10.0 # Bobot loss pose dinaikkan ke 10.0 untuk presisi kobj/keypoint tinggi
    close_mosaic = 5 # Nonaktifkan mosaic 5 epoch terakhir saja
    
    # Augmentasi Data Diperketat (Generalisasi Realistis)
    degrees = 25.0 # Rotasi sudut 25 derajat
    translate = 0.15 # Pergeseran translasi 15%
    scale = 0.5 # Skala zoom 50%
    shear = 5.0 # Distorsi geser 5 derajat
    fliplr = 0.5 # Flip horizontal
    mixup = 0.15 # Mixup 15% untuk pencampuran antar gambar
    erasing = 0.2 # Random erasing 20%
    
    output_base_dir = "models/trained_weights"
    os.makedirs(output_base_dir, exist_ok=True)

    # Cek ketersediaan CUDA GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ Menjalankan pada GPU: {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("⚠️ GPU CUDA tidak terdeteksi! Training akan berjalan di CPU.")

    for model_name in models:
        print(f"=== Memulai Pelatihan Teroptimasi: {model_name} ===")
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
            warmup_epochs=warmup_epochs,
            cos_lr=cos_lr,
            patience=patience,
            weight_decay=weight_decay,
            dropout=dropout,
            close_mosaic=close_mosaic,
            pose=pose_loss_weight,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            fliplr=fliplr,
            mixup=mixup,
            erasing=erasing,
            save_period=save_period,
            plots=plots,
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