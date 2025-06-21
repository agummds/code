#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf # Import TensorFlow untuk TFLite Interpreter
import time
import os
import math
import requests

# Pastikan lcd_display dan mqtt_client ada atau ganti dengan dummy jika tidak ada
try:
    from lcd_display import LCDDisplay
except ImportError:
    print("Warning: lcd_display not found. Skipping LCD display functionality.")
    class LCDDisplay:
        def __init__(self): pass
        def display_measurements(self, w, h): pass
        def cleanup(self): pass

try:
    from mqtt_client import MQTTClient
except ImportError:
    print("Warning: mqtt_client not found. Skipping MQTT functionality.")
    class MQTTClient:
        def __init__(self): pass
        def connect(self): pass
        def publish_measurement(self, height_cm, width_cm, confidence, class_id): pass
        def disconnect(self): pass


# Constants
# Pastikan path ini menunjuk ke file .tflite yang benar dari Colab Anda
MODEL_PATH = "mask_rcnn_model.tflite" 

# PENTING: Resolusi ini HARUS 640 karena model Anda dilatih dengan input 640x640
RESOLUTION_WIDTH = 640 
MODEL_INPUT_SIZE = RESOLUTION_WIDTH # Ukuran input model (tinggi/lebar)

FIXED_DISTANCE = 200     # cm
CAMERA_FOV = 30.9        # derajat, sesuaikan dengan FOV horizontal kamera kamu

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Initialize MQTT client
mqtt_client = MQTTClient()
mqtt_client.connect()

def hitung_pixel_to_cm(jarak_cm, fov_derajat, resolusi_horizontal):
    """Hitung nilai cm per pixel dari jarak dan FOV"""
    fov_rad = math.radians(fov_derajat / 2)
    lebar_cm = 2 * math.tan(fov_rad) * jarak_cm
    return lebar_cm / resolusi_horizontal

PIXEL_TO_CM = hitung_pixel_to_cm(FIXED_DISTANCE, CAMERA_FOV, RESOLUTION_WIDTH)


def load_tflite_model(model_path):
    """Memuat model TFLite dan menginisialisasi interpreter."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Model TFLite berhasil dimuat.")
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")
        
        # Verifikasi input shape yang diharapkan model
        model_input_shape = input_details[0]['shape']
        print(f"Model TFLite mengharapkan input shape: {model_input_shape}")
        if model_input_shape[1] != MODEL_INPUT_SIZE or model_input_shape[2] != MODEL_INPUT_SIZE:
            print(f"WARNING: MODEL_INPUT_SIZE ({MODEL_INPUT_SIZE}) tidak cocok dengan dimensi input model TFLite ({model_input_shape[1]}x{model_input_shape[2]}).")
            print("Pastikan MODEL_INPUT_SIZE di kode ini sesuai dengan model yang Anda konversi dari Colab (seharusnya 640).")

        return interpreter, input_details[0]['index'], output_details[0]['index']
    except Exception as e:
        print(f"Error memuat model TFLite: {e}")
        return None, None, None

def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def process_frame(frame, interpreter, input_idx, output_idx, lcd_display=None):
    """Memproses frame tunggal untuk segmentasi dan pengukuran tubuh menggunakan TFLite."""
    
    # 1. Resize dan padding gambar ke ukuran input model (640x640)
    frame_resized = resize_with_padding(frame, MODEL_INPUT_SIZE)
    
    # 2. KONVERSI BGR ke RGB (PENTING: Model Anda dilatih dengan RGB)
    frame_resized_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalisasi data (0-1) dan tambahkan dimensi batch
    input_data = frame_resized_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0) # Shape: (1, 640, 640, 3)
    
    # 4. Set tensor input dan jalankan inferensi TFLite
    interpreter.set_tensor(input_idx, input_data)
    interpreter.invoke()
    
    # Dapatkan output dari interpreter
    # Output model adalah (1, H, W, num_classes + 1), yaitu (1, 640, 640, 3)
    raw_predictions = interpreter.get_tensor(output_idx)
    
    # Ambil probabilitas untuk gambar pertama (indeks 0)
    predictions_per_pixel = raw_predictions[0] # Ini akan memiliki shape (640, 640, 3)

    # Debugging: print shape output mentah
    print(f"Shape output mentah dari TFLite setelah squeeze batch: {predictions_per_pixel.shape}")
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi untuk setiap piksel
    # Ini akan menghasilkan mask dengan nilai 0 (background), 1 (objek tubuh), 2 (objek lain/kosong)
    predicted_class_mask = np.argmax(predictions_per_pixel, axis=-1) # Shape: (640, 640)

    # Kita hanya tertarik pada Kelas 1 (Tubuh)
    # Buat mask biner di mana piksel yang termasuk Kelas 1 disetel ke 255, lainnya 0
    mask = (predicted_class_mask == 1).astype(np.uint8) * 255
    
    # Debugging: print persentase piksel terdeteksi setelah argmax untuk kelas 1
    detected_pixels_after_arg_max = np.sum(mask > 0)
    total_pixels_in_mask = mask.size
    print(f"Persentase piksel terdeteksi (kelas 1): {(detected_pixels_after_arg_max / total_pixels_in_mask) * 100:.2f}%")

    # Resize mask kembali ke ukuran frame asli dari kamera
    # Gunakan interpolation=cv2.INTER_NEAREST untuk mask biner agar tidak ada nilai abu-abu
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # PENTING: Pastikan Anda MENGHAPUS atau MENGOMENTARI baris ini jika Anda masih memilikinya!
    # Berdasarkan kode Colab Anda, mask tidak dibalikkan setelah argmax.
    # Jika mask Anda sekarang sudah 255 untuk objek dan 0 untuk background,
    # maka cv2.bitwise_not(mask) akan membalikkannya, yang mungkin tidak Anda inginkan.
    # mask = cv2.bitwise_not(mask) 
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_frame = frame.copy()
    
    confidence_percentage = 0.0 # Default value jika tidak ada deteksi
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Untuk perhitungan confidence: ambil probabilitas dari channel kelas 1 (tubuh)
        # di ukuran input model, lalu resize untuk rata-rata pada area objek yang terdeteksi
        # pada frame asli.
        object_confidence_map_resized = cv2.resize(
            predictions_per_pixel[:, :, 1], # Ambil channel probabilitas untuk kelas 1 (tubuh)
            (frame.shape[1], frame.shape[0]), 
            interpolation=cv2.INTER_LINEAR # Gunakan interpolasi linear untuk probabilitas
        )
        
        # Buat mask biner di ukuran frame asli untuk mengambil nilai confidence
        # Gunakan mask yang sudah di-resize dan di-morphology (variabel `mask`)
        if np.sum(mask > 0) > 0:
            confidence_percentage = np.mean(object_confidence_map_resized[mask > 0]) * 100
        else:
            confidence_percentage = 0.0

        # Visualisasi mask hijau di atas objek
        mask_color = np.zeros_like(frame)
        mask_color[mask > 0] = [0, 255, 0] # Objek akan berwarna hijau (BGR)
        result_frame = cv2.addWeighted(result_frame, 0.7, mask_color, 0.3, 0)
        
        # Gambar bounding box (merah)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Hitung pengukuran dalam cm
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        # Kirim ke MQTT (konversi ke float standar Python untuk serialisasi JSON)
        mqtt_client.publish_measurement(
            height_cm=float(height_cm), 
            width_cm=float(width_cm),   
            confidence=float(confidence_percentage / 100.0), 
            class_id=1 # Class ID untuk tubuh
        )
        
        # Tampilkan teks pengukuran dan confidence
        measurements = f"W: {width_cm:.1f}cm H: {height_cm:.1f}cm"
        confidence_text = f"Conf: {confidence_percentage:.1f}%"
        
        # Posisi teks: pengukuran di atas bbox, confidence di bawah bbox atau di samping
        text_size_meas = cv2.getTextSize(measurements, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_frame, (x, y - text_size_meas[1] - 10), (x + text_size_meas[0], y), (0, 0, 0), -1)
        cv2.putText(result_frame, measurements, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Kuning
        
        text_size_conf = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_frame, (x, y + h + 5), (x + text_size_conf[0], y + h + text_size_conf[1] + 15), (0, 0, 0), -1)
        cv2.putText(result_frame, confidence_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Biru Muda
        
        print(f"\nPengukuran tubuh pada jarak {FIXED_DISTANCE}cm:")
        print(f"Lebar: {width_cm:.1f} cm")
        print(f"Tinggi: {height_cm:.1f} cm")
        print(f"Dimensi piksel: {w}x{h}")
        print(f"Persentase Keyakinan Objek (Tubuh): {confidence_percentage:.2f}%")
        
        if lcd_display is not None:
            lcd_display.display_measurements(width_cm, height_cm)
            # Anda mungkin ingin juga menampilkan confidence di LCD jika ada baris tambahan
            # atau menggabungkan dengan tinggi/lebar.
    else:
        print("\nTidak ada objek terdeteksi.")
        if lcd_display is not None:
            lcd_display.display_measurements(0, 0) # Atau tampilkan pesan "Tidak terdeteksi"

    return result_frame

def main():
    # Pastikan model TFLite ada
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File model tidak ditemukan di '{MODEL_PATH}'")
        return

    # Muat model TFLite
    interpreter, input_idx, output_idx = load_tflite_model(MODEL_PATH)
    if interpreter is None:
        return

    # Inisialisasi LCD display
    try:
        lcd = LCDDisplay()
        print("LCD display berhasil diinisialisasi")
    except Exception as e:
        print(f"Error menginisialisasi LCD display: {e}")
        lcd = None

    # Inisialisasi kamera
    print("Menghidupkan kamera...")
    # Coba berbagai backend kamera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # V4L2 sering kali lebih baik di RPi
    if not cap.isOpened():
        cap = cv2.VideoCapture(0) # Fallback ke default
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka. Pastikan kamera terhubung dan diaktifkan.")
        return

    # Set resolusi kamera. Ini adalah resolusi frame yang diambil, bukan input model.
    # Disarankan set resolusi yang mendekati rasio aspek model atau kamera Anda.
    # Misalnya, jika model input 640x640, Anda bisa mencoba mengatur kamera ke 640x480 atau 640x640 (jika didukung).
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Tekan SPASI untuk mengambil gambar, atau 'q' untuk keluar.")
    frame = None
    while True:
        ret, preview = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        # Rotasi gambar jika kamera terpasang vertikal (seperti di gambar Anda sebelumnya)
        preview = cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Preview Kamera", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # tombol SPASI ditekan
            frame = preview.copy()
            print("Gambar diambil.")
            break
        elif key == ord('q'):
            print("Keluar tanpa mengambil gambar.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyWindow("Preview Kamera")

    if frame is not None:
        print("Memproses gambar...")
        result = process_frame(frame, interpreter, input_idx, output_idx, lcd)
        cv2.imshow("Hasil Deteksi", result)
        print("Tekan tombol apa saja untuk keluar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if lcd is not None:
        lcd.cleanup()
    mqtt_client.disconnect()

if __name__ == "__main__":
    main()
