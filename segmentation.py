import cv2
import numpy as np
import tensorflow as tf
import time
import os
import math
import requests
from lcd_display import LCDDisplay
from mqtt_client import MQTTClient

# Constants
MODEL_URL = "https://raw.githubusercontent.com/agummds/Mask-RCNN-TA/master/model.tflite"
MODEL_PATH = "model.tflite"
FIXED_DISTANCE = 150  # cm
CAMERA_FOV = 70  # derajat, sesuaikan dengan FOV horizontal kamera kamu
RESOLUTION_WIDTH = 640  # sesuaikan dengan resolusi input model

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
MODEL_INPUT_SIZE = 640

def download_model():
    """Download the TFLite model if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    return True

def load_model():
    """Load and initialize TFLite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading model: {e}")
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

def process_frame(frame, interpreter, input_details, output_details, lcd_display=None):
    """Process a single frame for body segmentation and measurement"""
    
    input_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    frame_resized = resize_with_padding(frame, MODEL_INPUT_SIZE)

    
    input_data = frame_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    mask = interpreter.get_tensor(output_details[0]['index'])[0]
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = cv2.bitwise_not(mask)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        mask_color = np.zeros_like(frame)
        mask_color[mask > 0] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.7, mask_color, 0.3, 0)
        
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        mqtt_client.publish_measurement(
            height_cm=height_cm,
            width_cm=width_cm,
            confidence=1.0,
            class_id=1
        )
        
        measurements = f"W: {width_cm:.1f}cm H: {height_cm:.1f}cm"
        text_size = cv2.getTextSize(measurements, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result, (x, y - text_size[1] - 10), 
                      (x + text_size[0], y), (0, 0, 0), -1)
        cv2.putText(result, measurements, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        print(f"\nBody measurements at {FIXED_DISTANCE}cm distance:")
        print(f"Width: {width_cm:.1f} cm")
        print(f"Height: {height_cm:.1f} cm")
        print(f"Pixel dimensions: {w}x{h}")
        
        if lcd_display is not None:
            lcd_display.display_measurements(width_cm, height_cm)
    
    return result

def main():
    # Download dan load model
    if not download_model():
        return
    interpreter, input_details, output_details = load_model()
    if interpreter is None:
        return

    # Inisialisasi LCD display
    try:
        lcd = LCDDisplay()
        print("LCD display initialized successfully")
    except Exception as e:
        print(f"Error initializing LCD display: {e}")
        lcd = None

    # Inisialisasi kamera
    print("Menghidupkan kamera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka.")
        return

    print("Tekan SPASI untuk mengambil gambar, atau 'q' untuk keluar.")

    frame = None
    while True:
        ret, preview = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

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
        result = process_frame(frame, interpreter, input_details, output_details, lcd)
        cv2.imshow("Hasil Deteksi", result)
        print("Tekan tombol apa saja untuk keluar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if lcd is not None:
        lcd.cleanup()
    mqtt_client.disconnect()


if __name__ == "__main__":
    main()
