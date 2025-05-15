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
    
    # Resize to match model input size exactly
    # This is necessary because camera might not support exact 640x640 resolution
    input_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    frame_resized = cv2.resize(frame, input_size)
    
    # Prepare input tensor
    input_data = frame_resized.astype(np.float32) / 255.0
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
    
    # Resize mask back to original frame size for visualization
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Use simpler morphological operations
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization
    result = frame.copy()
    
    if contours and len(contours) > 0:
        try:
            # Get largest contour (assuming it's the body)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Calculate measurements
            width_cm = w * PIXEL_TO_CM
            height_cm = h * PIXEL_TO_CM
            
            # Send measurements through MQTT
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
        except Exception as e:
            print(f"Error processing contours: {e}")
    
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
    
    # Check input shape expected by model
    input_shape = input_details[0]['shape']
    print(f"Model expects input shape: {input_shape}")
    
    # Initialize LCD display
    try:
        lcd = LCDDisplay()
        print("LCD display initialized successfully")
    except Exception as e:
        print(f"Error initializing LCD display: {e}")
        lcd = None
    
    # Try different capture methods for better compatibility
    print("Initializing camera...")
    cap = None
    
    # Try different backends
    for backend in [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"Successfully opened camera with backend {backend}")
                break
        except Exception:
            continue
    
    # If all backends failed, try default
    if cap is None or not cap.isOpened():
        print("Trying default camera...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties - try to get closest to model input size
    # Note: Camera may not support exactly 640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MODEL_INPUT_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MODEL_INPUT_SIZE)
    
    # Try to reduce frame rate to save processing power
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Get actual camera resolution (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera actual resolution: {actual_width}x{actual_height}")
    print(f"Processing every {PROCESS_EVERY_N_FRAMES}th frame")
    print("Press 'q' to quit")
    
    # Quick test frame
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        return
    
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
                try:
                    # Process frame
                    result, fps = process_frame(frame, interpreter, input_details, output_details, lcd)
                    fps_avg = 0.9 * fps_avg + 0.1 * fps  # Smooth FPS calculation
                    
                    # Show result
                    cv2.imshow("Segmentation and Measurement", result)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Show original frame if processing fails
                    cv2.imshow("Segmentation and Measurement", frame)
            else:
                # For skipped frames, just show the original frame with FPS
                try:
                    copy_frame = frame.copy()
                    cv2.putText(copy_frame, f"FPS: {fps_avg:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("Segmentation and Measurement", copy_frame)
                except Exception as e:
                    print(f"Error displaying frame: {e}")
            
            # Break on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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