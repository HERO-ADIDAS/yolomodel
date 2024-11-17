from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
import io
import numpy as np
import cv2
from PIL import Image as PILImage
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class ObjectDetectionSystem:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectDetectionSystem, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            # Initialize YOLO model only when needed
            self.initialized = True
            self._yolo_model = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO('yolov8n.pt')
        return self._yolo_model
    
    def detect_objects(self, image_base64: str, confidence_threshold: float = 0.5):
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
detector = ObjectDetectionSystem()

@app.post("/api/detect")
async def detect_objects(request: ImageRequest):
    try:
        result_image = detector.detect_objects(request.image)
        return {"processed_image": result_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}