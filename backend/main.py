"""
Indian Sign Language Detection & Recognition System - FastAPI Backend Server
"""

import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.api.routes import router

app = FastAPI(
    title="Real-Time Indian Sign Language Detection API",
    description="Real-Time ISL detection and recognition using Computer Vision, YOLO, and PyTorch Deep Learning",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(router, prefix="/api")

@app.get("/api/health")
def health_check():
    import torch
    import cv2
    import ultralytics
    return {
        "status": "healthy",
        "service": "ISL Detection & Recognition Engine",
        "versions": {
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "opencv": cv2.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "device": "cpu",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
