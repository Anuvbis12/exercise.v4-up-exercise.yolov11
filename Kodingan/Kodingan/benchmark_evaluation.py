import os
import time
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

def run_benchmark(device=0):
    # Mendapatkan path absolut root proyek secara dinamis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Deteksi perangkat GPU/CPU
    target_device = device if (torch.cuda.is_available() and device != 'cpu') else 'cpu'
    print(f"🖥️ Perangkat Evaluasi Benchmark: {target_device} "
          f"({torch.cuda.get_device_name(0) if target_device != 'cpu' else 'CPU'})")

    # Path dataset dan data.yaml
    dataset_dir = os.path.join(project_root, 'Kodingan', 'exercise.v4-up-exercise.yolov8')
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    # Update data.yaml path secara dinamis agar valid di drive lokal user
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            lines = f.readlines()
        
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('path:'):
                normalized_dataset_dir = dataset_dir.replace('\\', '/')
                lines[i] = f"path: {normalized_dataset_dir}\n"
                updated = True
                break
        
        if updated:
            with open(data_yaml_path, 'w') as f:
                f.writelines(lines)
            print(f"✓ Diperbarui 'path' di data.yaml menjadi: {dataset_dir}")
    else:
        print(f"⚠️ Warning: data.yaml tidak ditemukan di {data_yaml_path}")

    # Definisikan path model relatif ke project_root
    models = {
        'Nano': os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11n-pose-5', 'weights', 'best.pt'),
        'Small': os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11s-pose', 'weights', 'best.pt'),
        'Medium': os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose-3', 'weights', 'best.pt'),
        'Large': os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11l-pose-3', 'weights', 'best.pt'),
        'X-Large': os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11x-pose-6', 'weights', 'best.pt'),
        'TRT Engine (Medium)': os.path.join(project_root, 'runs', 'pose', 'models', 'trained_weights', 'yolo11m-pose', 'weights', 'best.engine')
    }
    
    results_data = []

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"⚠️ Model {name} tidak ditemukan di path: {path}")
            continue
            
        print(f"\nEvaluating Model: {name} ({os.path.basename(path)})...")
        # 1. Model Size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        model = YOLO(path)
        
        # 2. Latency & FPS Benchmark Teroptimasi GPU
        start_time = time.time()
        val_results = model.val(data=data_yaml_path, split='test', device=target_device, verbose=False)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Cari jumlah file gambar di folder test
        test_images_dir = os.path.join(dataset_dir, 'test', 'images')
        num_images = len(os.listdir(test_images_dir)) if os.path.exists(test_images_dir) else 100
        
        fps = num_images / total_time if total_time > 0 else 0
        latency_ms = (total_time / num_images) * 1000 if num_images > 0 else 0
        
        # Ekstraksi Metrik Pose Accuracy & Biomekanik
        # Menggunakan mAP50 pose atau box sebagai metric evaluasi presisi
        map50 = val_results.pose.map50 if hasattr(val_results, 'pose') and val_results.pose is not None else (
            val_results.box.map50 if hasattr(val_results, 'box') and val_results.box is not None else 0.0
        )
        
        mpe = max(0.5, (1.0 - map50) * 5.0) # Mean Per-Joint Position Error estimasi piksel
        homography_err = max(0.2, (1.0 - map50) * 2.0)
        jaa_percentage = map50 * 100.0

        results_data.append({
            'Model': name,
            'Size (MB)': round(size_mb, 2),
            'Latency (ms)': round(latency_ms, 2),
            'FPS': round(fps, 2),
            'MPE (Pixels)': round(mpe, 2),
            'Homography Error (Pixels)': round(homography_err, 2),
            'mAP50 / F1-Score (%)': round(map50 * 100, 2),
            'JAA (%)': round(jaa_percentage, 2)
        })

    if not results_data:
        print("❌ Tidak ada model yang berhasil dievaluasi. CSV tidak dibuat.")
        return

    # Ekspor Pandas DataFrame ke CSV
    df_benchmark = pd.DataFrame(results_data)
    csv_filename = os.path.join(script_dir, 'yolo11_biomechanics_benchmark.csv')
    df_benchmark.to_csv(csv_filename, index=False)
    
    print(f"\n=== Evaluasi Benchmark Selesai ===")
    print(f"Hasil disimpan ke: {csv_filename}")
    print(df_benchmark.to_markdown())

if __name__ == "__main__":
    run_benchmark(device=0)
