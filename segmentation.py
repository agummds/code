import cv2
import numpy as np
import tensorflow as tf
import time
import os
import requests
from lcd_display import LCDDisplay
from mqtt_client import MQTTClient

# Constants
MODEL_URL = "https://raw.githubusercontent.com/agummds/Mask-RCNN-TA/master/model.tflite"
MODEL_PATH = "model.tflite"
FIXED_DISTANCE = 150  # cm
PIXEL_TO_CM = 0.187  # cm per pixel
MODEL_INPUT_SIZE = 640  # Keeping original model size
PROCESS_EVERY_N_FRAMES = 3  # Only process every 3rd frame

# Initialize MQTT client
mqtt_client = MQTTClient()
mqtt_client.connect()

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
        # Set num_threads for better performance on RPi
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=4  # Adjust based on your RPi's core count
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def process_frame(frame, interpreter, input_details, output_details, lcd_display=None):
    """Process a single frame for body segmentation and measurement"""
    start_time = time.time()
    
    # Prepare input tensor (no resizing, just normalization)
    input_data = frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get segmentation mask
    mask = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Process mask - ensure it's single channel binary mask
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Take first channel only
    
    # Convert to binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Use simpler morphological operations
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization
    result = frame.copy()
    
    if contours:
        # Get largest contour (assuming it's the body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw segmentation mask overlay (optional - comment out if too slow)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_rgb[:, :, 0:2] = 0  # Keep only green channel
        result = cv2.addWeighted(result, 0.7, mask_rgb, 0.3, 0)
        
        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Calculate measurements
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        # Send measurements through MQTT (less frequently)
        mqtt_client.publish_measurement(
            height_cm=height_cm,
            width_cm=width_cm,
            confidence=1.0,
            class_id=1
        )
        
        # Add measurements to frame with simpler text
        measurements = f"W: {width_cm:.1f}cm H: {height_cm:.1f}cm"
        cv2.putText(result, measurements, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Update LCD display if available
        if lcd_display is not None:
            lcd_display.display_measurements(width_cm, height_cm)
    
    # Calculate and display FPS
    process_time = time.time() - start_time
    fps = 1.0 / process_time if process_time > 0 else 0
    cv2.putText(result, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return result, fps

def main():
    # Download and load model
    if not download_model():
        return
    
    interpreter, input_details, output_details = load_model()
    if interpreter is None:
        return
    
    # Initialize LCD display
    try:
        lcd = LCDDisplay()
        print("LCD display initialized successfully")
    except Exception as e:
        print(f"Error initializing LCD display: {e}")
        lcd = None
    
    # Initialize camera with low resolution directly matching model input
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties - lower resolution to match model
    # Set to 640x640 or closest available resolution to avoid resizing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MODEL_INPUT_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MODEL_INPUT_SIZE)
    # Try to reduce frame rate to save processing power
    cap.set(cv2.CAP_PROP_FPS, 15)
        
    # Quick test frame
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        return
        
    # Get actual camera resolution (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera initialized at resolution: {actual_width}x{actual_height}")
    print(f"Processing every {PROCESS_EVERY_N_FRAMES}th frame")
    print("Press 'q' to quit")
    
    frame_count = 0
    fps_avg = 0
    last_process_time = time.time()
    
    try:
        while True:
            # Limit frame rate by adding small delay
            if time.time() - last_process_time < 0.01:  # ~100fps max
                time.sleep(0.01)
                
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            frame_count += 1
            last_process_time = time.time()
            
            # Only process every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Process frame
                result, fps = process_frame(frame, interpreter, input_details, output_details, lcd)
                fps_avg = 0.9 * fps_avg + 0.1 * fps  # Smooth FPS calculation
                
                # Show result
                cv2.imshow("Segmentation and Measurement", result)
            else:
                # For skipped frames, just show the original frame with FPS
                copy_frame = frame.copy()
                cv2.putText(copy_frame, f"FPS: {fps_avg:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Segmentation and Measurement", copy_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if lcd is not None:
            lcd.cleanup()
        mqtt_client.disconnect()  # Disconnect MQTT client

if __name__ == "__main__":
    main()