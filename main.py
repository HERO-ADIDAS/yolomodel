from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import cv2
from PIL import Image as PILImage
from ultralytics import YOLO
import os

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

# Global model instance
model = None

def load_model():
    global model
    if model is None:
        try:
            # Using the smallest YOLO model
            model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Error loading model")
    return model

@app.post("/api/detect")
async def detect_objects(request: ImageRequest):
    try:
        # Load model on first request
        yolo_model = load_model()
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            pil_image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(pil_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Run inference with smaller confidence threshold
        results = yolo_model(image_np, conf=0.25)[0]  # Reduced confidence threshold
        
        # Process detections
        output_image = image_np.copy()
        
        # Get detections
        boxes = results.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            # Draw rectangle
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(output_image, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to base64
        pil_output = PILImage.fromarray(output_image)
        buffer = io.BytesIO()
        pil_output.save(buffer, format="JPEG", quality=95)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {"processed_image": encoded_image}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}