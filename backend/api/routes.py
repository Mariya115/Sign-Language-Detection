"""
FastAPI REST and WebSocket Routes for Real-Time ISL System
"""

import time
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.camera.stream_manager import CameraStreamManager
from backend.services.text_to_speech import TextToSpeechEngine

router = APIRouter()

# Global instances
stream_manager = CameraStreamManager()
tts_engine = TextToSpeechEngine()

# Recognition event history (in-memory ring buffer)
recognition_history: List[Dict[str, Any]] = []
MAX_HISTORY_LEN = 100


class CameraStartRequest(BaseModel):
    camera_id: int = 0
    frame_skip: int = 2


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"  # "en" or "kn"


class TextActionRequest(BaseModel):
    action: str  # "clear", "backspace", "space"


@router.post("/camera/start")
def start_camera(req: CameraStartRequest = CameraStartRequest()):
    if req.frame_skip < 1:
        raise HTTPException(status_code=400, detail="frame_skip must be at least 1")
    stream_manager.frame_skip = req.frame_skip
    success = stream_manager.start(camera_id=req.camera_id)
    if not success:
        raise HTTPException(status_code=503, detail="Unable to start camera stream")
    return {
        "status": "started",
        "camera_id": req.camera_id,
        "is_running": stream_manager.is_running,
        "result": stream_manager.get_latest_result(),
    }


@router.post("/camera/stop")
def stop_camera():
    stream_manager.stop()
    return {"status": "stopped"}


@router.get("/camera/status")
def get_camera_status():
    result = stream_manager.get_latest_result()
    return {
        "is_running": stream_manager.is_running,
        "camera_id": stream_manager.camera_id,
        "use_synthetic_source": stream_manager.use_synthetic_source,
        "result": result,
    }


@router.get("/camera/stream")
def video_stream():
    """MJPEG continuous video stream endpoint for frontend <img src>."""
    return StreamingResponse(
        stream_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/model/info")
def get_model_info():
    """Return model architecture, supported ISL classes, and evaluation metrics."""
    engine = stream_manager.engine
    return {
        "model_architecture": getattr(engine.model, "__class__", type("ISLModel", (), {})).__name__,
        "classifier_checkpoint": engine.classifier_checkpoint,
        "num_classes": engine.num_classes,
        "classes": [c.upper() for c in engine.classes],
        "hand_detector": "YOLOv8n hand detector",
        "keypoint_model": "MediaPipe Hands 21-point landmark model",
        "keypoint_model_available": stream_manager.engine.detector.mp_hands is not None,
        "gesture_model": "Five-class Kannada landmark classifier",
        "gesture_model_available": stream_manager.engine.gesture_model_available,
        "gesture_classes": ["ನಮಸ್ಕಾರ", "ಹೌದು", "ಇಲ್ಲ", "ಸರಿ", "ಮುಗಿದಿದೆ"],
        "confidence_threshold": engine.conf_threshold,
        "smoothing_window": engine.smoothing_window,
        "input_resolution": f"{engine.target_size}x{engine.target_size} RGB",
        "dataset_name": "Indian Sign Language (ISL) Alphabets & Gestures",
        "supported_features": [
            "YOLO Hand Localization",
            "MediaPipe 21-point hand landmark extraction",
            "Single & Multi-Hand Crop",
            "Deep Learning Classification",
            "Confidence Gating",
            "Temporal Majority Smoothing",
            "English & Kannada TTS",
            "Sentence Construction",
            "Kannada alphabet labels",
            "Five static landmark gestures: open palm, thumbs up, thumbs down, OK sign, closed fist",
        ],
    }


@router.post("/text/action")
def text_action(req: TextActionRequest):
    sb = stream_manager.sentence_builder
    if req.action == "clear":
        state = sb.clear()
    elif req.action == "backspace":
        state = sb.backspace()
    elif req.action == "space":
        state = sb.add_space()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    return {"status": "ok", "state": state}


@router.post("/tts")
def synthesize_speech(req: TTSRequest):
    audio_b64 = tts_engine.synthesize_audio_base64(req.text, lang=req.lang)
    if not audio_b64:
        raise HTTPException(status_code=500, detail="Audio synthesis failed")
    return {
        "status": "success",
        "text": req.text,
        "lang": req.lang,
        "audio_base64": audio_b64,
    }


@router.get("/history")
def get_recognition_history():
    return {
        "total": len(recognition_history),
        "history": list(reversed(recognition_history[-50:])),
    }


@router.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    Real-time WebSocket endpoint pushing live predictions at ~30Hz.
    """
    await websocket.accept()
    last_sent_prediction = ""
    try:
        while True:
            result = stream_manager.get_latest_result()
            payload = {
                "timestamp": time.time(),
                "is_running": stream_manager.is_running,
                "prediction": result.get("prediction", "No Hand"),
                "confidence": result.get("confidence", 0.0),
                "raw_confidence": result.get("raw_confidence", 0.0),
                "fps": result.get("fps", 0.0),
                "latency_ms": result.get("latency_ms", 0.0),
                "hand_detected": result.get("hand_detected", False),
                "landmarks_detected": result.get("landmarks_detected", False),
                "landmark_count": result.get("landmark_count", 0),
                "gesture_recognized": result.get("gesture_recognized", False),
                "predicted_class": result.get("predicted_class"),
                "kannada_label": result.get("kannada_label"),
                "english_label": result.get("english_label"),
                "status": result.get("status", "unknown"),
                "num_hands": result.get("num_hands", 0),
                "hand_detection_confidence": result.get("hand_detection_confidence", 0.0),
                "boxes": result.get("boxes", []),
                "landmarks": result.get("landmarks", []),
                "kannada_prediction": result.get("kannada_label"),
                "gesture": result.get("gesture"),
                "probabilities": result.get("probabilities", {}),
                "sentence": stream_manager.sentence_builder.get_sentence(),
            }

            # Record stable history events
            curr_pred = payload["prediction"]
            if result.get("status") == "recognized" and curr_pred:
                if curr_pred != last_sent_prediction:
                    recognition_history.append({
                        "id": len(recognition_history) + 1,
                        "sign": curr_pred,
                        "confidence": payload["confidence"],
                        "timestamp": time.strftime("%H:%M:%S"),
                        "num_hands": payload["num_hands"],
                    })
                    if len(recognition_history) > MAX_HISTORY_LEN:
                        recognition_history.pop(0)
                    last_sent_prediction = curr_pred

            await websocket.send_json(payload)
            await asyncio.sleep(0.04)  # ~25-30 updates/sec
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Disconnected: {e}")
