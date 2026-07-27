import os
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO

def run_benchmark():
    models = {
        'Nano': 'models/trained_weights/yolo11n-pose/weights/best.pt',
        #'Small': 'models/trained_weights/yolo11s-pose/weights/best.pt',
        #'Medium': 'models/trained_weights/yolo11m-pose/weights/best.pt',
        #'Large': 'models/trained_weights/yolo11l-pose/weights/best.pt',
        #'X-Large': 'models/trained_weights/yolo11x-pose/weights/best.pt'
    }
    
    results_data = []

    for name, path in models.items():
        if not os.path.exists(path):
            continue
            
        # 1. Model Size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        model = YOLO(path)
        
        # 2. Latency & FPS 
        # (Simulasi menggunakan validasi YOLO untuk iterasi cepat)
        start_time = time.time()
        val_results = model.val(data='dataset/data.yaml', split='test', device=0)
        end_time = time.time()
        
        total_time = end_time - start_time
        num_images = 100 # Ganti dengan jumlah gambar test Anda
        fps = num_images / total_time
        latency_ms = (total_time / num_images) * 1000
        
        # 3, 4, 5, 6. Ekstraksi Metrik Lainnya
        # Implementasikan logika ekstraksi untuk MPE, Sampson's Error, F1, JAA
        # Nilai di bawah ini diisi dengan dummy sebagai contoh struktur evaluasi DataFrame
        mpe = np.random.uniform(1.5, 3.5) 
        homography_err = np.random.uniform(0.5, 1.2)
        f1_score = val_results.box.map50 # Memanfaatkan map50 model pose sebagai representasi akurasi dasar
        jaa_percentage = np.random.uniform(85.0, 98.0)

        results_data.append({
            'Model': name,
            'Size (MB)': round(size_mb, 2),
            'Latency (ms)': round(latency_ms, 2),
            'FPS': round(fps, 2),
            'MPE (Pixels)': round(mpe, 2),
            'Homography Error (Pixels)': round(homography_err, 2),
            'F1-Score (%)': round(f1_score * 100, 2),
            'JAA (%)': round(jaa_percentage, 2)
        })

    # Ekspor Pandas DataFrame ke CSV
    df_benchmark = pd.DataFrame(results_data)
    csv_filename = 'yolo11_biomechanics_benchmark.csv'
    df_benchmark.to_csv(csv_filename, index=False)
    
    print(f"\n=== Evaluasi Benchmark Selesai ===")
    print(df_benchmark.to_markdown())

if __name__ == "__main__":
    run_benchmark()