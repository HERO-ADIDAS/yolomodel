import io
import base64
import numpy as np
import cv2
from PIL import Image as PILImage
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class ObjectDetectionSystem:
    def __init__(self, model_path='yolov8n.pt'):
        # Initialize YOLO model
        self.yolo_model = YOLO(model_path)

    def detect_objects(self, image_base64: str, confidence_threshold: float = 0.5):
        """
        Perform object detection on base64 encoded image
        
        Args:
            image_base64 (str): Input image as base64 string
            confidence_threshold (float): Minimum confidence for detections
            
        Returns:
            str: Output image as base64 string with detected objects pk
        """
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            pil_image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(pil_image)
            
            # Run YOLO model
            results = self.yolo_model(image_np)
            
            # Process detections
            output_image = image_np.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    if confidence > confidence_threshold:
                        # Draw bounding box
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        label = f'{class_name}: {confidence:.2f}'
                        cv2.putText(output_image, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert back to base64
            output_pil = PILImage.fromarray(output_image)
            buffered = io.BytesIO()
            output_pil.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return encoded_image
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize the detection system
detector = ObjectDetectionSystem('yolov8m.pt')  # Replace with your model path

@app.post("/detect")
async def detect_objects(request: ImageRequest):
    try:
        # Process the image
        result_image = detector.detect_objects(request.image)
        return {"processed_image": result_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For testing the endpoint
@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}

"""
Save this code as 'main.py' and run it from the command line using:
    uvicorn main:app --host 0.0.0.0 --port 8000

Example usage with Python requests:

import requests

url = "http://localhost:8000/detect"
payload = {
    "image": "your_base64_string_here"
}
response = requests.post(url, json=payload)
result = response.json()
processed_image = result["processed_image"]
"""