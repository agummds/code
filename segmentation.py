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
MODEL_INPUT_SIZE = 640

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
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def process_frame(frame, interpreter, input_details, output_details, lcd_display=None):
    """Process a single frame for body segmentation and measurement"""
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for visualization
    
    # Resize frame to model input size
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
    
    # Convert to binary mask and clean up
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Invert the mask to highlight the body instead of background
    mask = cv2.bitwise_not(mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization using grayscale frame
    result = frame_gray.copy()
    
    if contours:
        # Get largest contour (assuming it's the body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw segmentation mask with transparency
        mask_color = np.zeros_like(frame_gray)
        mask_color[mask > 0] = [0, 255, 0]  # Green mask
        result = cv2.addWeighted(result, 0.7, mask_color, 0.3, 0)
        
        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Calculate measurements
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        # Send measurements through MQTT
        mqtt_client.publish_measurement(
            height_cm=height_cm,
            width_cm=width_cm,
            confidence=1.0,  # Since we're using the largest contour
            class_id=1  # Assuming 1 for body
        )
        
        # Add measurements to frame with background for better visibility
        measurements = f"W: {width_cm:.1f}cm H: {height_cm:.1f}cm"
        text_size = cv2.getTextSize(measurements, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result, (x, y - text_size[1] - 10), 
                     (x + text_size[0], y), (0, 0, 0), -1)
        cv2.putText(result, measurements, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Print measurements to terminal
        print(f"\nBody measurements at {FIXED_DISTANCE}cm distance:")
        print(f"Width: {width_cm:.1f} cm")
        print(f"Height: {height_cm:.1f} cm")
        print(f"Pixel dimensions: {w}x{h}")
        
        # Update LCD display if available
        if lcd_display is not None:
            lcd_display.display_measurements(width_cm, height_cm)
    
    return result

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
    
    # Initialize camera with minimal settings
    print("Initializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow first for faster initialization
    
    if not cap.isOpened():
        print("DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Quick test frame
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        return
    
    print("\nCamera initialized successfully")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            result = process_frame(frame, interpreter, input_details, output_details, lcd)
            
            # Show result
            cv2.imshow("Segmentation and Measurement", result)
            
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