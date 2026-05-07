"""
FastAPI Backend for Intracranial Aneurysm Detection
Using MIC-DKFZ 7th Place Solution (nnU-Net)

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import uuid
import time
import asyncio
import queue

# Import inference module
from inference import get_inference_engine, check_model_status, LOCATION_LABELS

# Initialize FastAPI app
app = FastAPI(
    title="Intracranial Aneurysm Detection API",
    description="API for detecting intracranial aneurysms using pretrained nnU-Net model",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modular routers ───────────────────────────────────────────
from dataset_router import router as dataset_router
app.include_router(dataset_router)
# ─────────────────────────────────────────────────────────────



# Pydantic models
class PredictionResult(BaseModel):
    location: str
    probability: float
    detected: bool
    coordinates: Optional[Dict[str, int]] = None  # {'x': int, 'y': int, 'z': int}
    detailed_coordinates: Optional[List[Dict[str, Any]]] = None # [{'x': int, 'y': int, 'z': int, 'prob': float}]
    slice_number: Optional[int] = None


class AnalysisResponse(BaseModel):
    id: str
    status: str
    predictions: List[PredictionResult]
    overall_risk: str
    confidence: float
    processing_time: float
    model_loaded: bool = False
    image_base64: Optional[str] = None
    slice_index: Optional[int] = None
    bbox: Optional[List[int]] = None
    modality: Optional[str] = None
    slice_images: Optional[List[Dict[str, Any]]] = None



class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_exists: bool
    download_url: str


class ModelStatusResponse(BaseModel):
    model_loaded: bool
    checkpoint_exists: bool
    checkpoint_path: str
    download_url: str


# Store for analysis results
analysis_store: Dict[str, Any] = {}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Global log queue to bridge sync code -> async websocket
import queue
import asyncio
log_queue = queue.Queue()

def log_to_frontend(message: str):
    """Callback to push logs to frontend"""
    print(message) # Keep console output
    log_queue.put(message)

# Background task to stream logs
async def log_streamer():
    while True:
        try:
            # Non-blocking get
            msg = log_queue.get_nowait()
            await manager.broadcast(msg)
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Log streamer error: {e}")
            await asyncio.sleep(1)



@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup."""
    print("Initializing inference engine...")
    engine = get_inference_engine()
    status = check_model_status()
    print(f"Model status: {status}")
    
    # Start log streamer
    asyncio.create_task(log_streamer())


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    status = check_model_status()
    return HealthResponse(
        status="healthy",
        model_loaded=status["model_loaded"],
        checkpoint_exists=status["checkpoint_exists"],
        download_url=status["download_url"],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status = check_model_status()
    return HealthResponse(
        status="healthy",
        model_loaded=status["model_loaded"],
        checkpoint_exists=status["checkpoint_exists"],
        download_url=status["download_url"],
    )


@app.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """Get detailed model status."""
    return check_model_status()


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_dicom(
    files: List[UploadFile] = File(...),
    modality: str = "CTA",
):
    """
    Analyze DICOM files for aneurysm detection.
    Uses the pretrained nnU-Net model if available, otherwise simulation.
    """
    start_time = time.time()
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())[:8]
    
    # Create temporary directory for DICOM files
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        print(f"DEBUG: Received analyzing request with {len(files)} files.")
        
        # Save uploaded files
        print(f"DEBUG: Saving files to temporary directory {temp_dir}...")
        for f in files:
            file_path = temp_dir / f.filename
            with open(file_path, 'wb') as buffer:
                content = await f.read()
                buffer.write(content)
        
        print("DEBUG: All files saved successfully.")

        # Get inference engine
        engine = get_inference_engine()
        
        # Run prediction
        print("DEBUG: Calling engine.predict() in threadpool...")
        from fastapi.concurrency import run_in_threadpool
        
        # Pass the log callback
        result = await run_in_threadpool(engine.predict, temp_dir, log_to_frontend)
        print("DEBUG: Prediction returned successfully.")
        
        # Create predictions list
        predictions = []
        threshold = 0.5
        
        for location in LOCATION_LABELS:
            prob = result["probabilities"].get(location, 0.1)
            coords = result.get("coordinates", {}).get(location)
            details = result.get("detailed_coordinates", {}).get(location)
            
            predictions.append(PredictionResult(
                location=location,
                probability=float(prob),
                detected=bool(prob > threshold),
                coordinates=coords,
                detailed_coordinates=details,
                slice_number=coords.get("z") if coords else None
            ))
        
        # Get risk level
        risk = result["risk_level"]
        confidence = result["max_probability"]
        
        processing_time = time.time() - start_time
        
        response = AnalysisResponse(
            id=analysis_id,
            status="completed",
            predictions=predictions,
            overall_risk=risk,
            confidence=float(confidence),
            processing_time=processing_time,
            model_loaded=result.get("model_loaded", False),
            image_base64=result.get("image_base64"),
            slice_index=result.get("slice_index"),
            bbox=result.get("bbox"),
            modality=result.get("modality"),
            slice_images=result.get("slice_images"),
        )
        
        analysis_store[analysis_id] = response
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis result by ID."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_store[analysis_id]


@app.get("/locations")
async def get_locations():
    """Get list of anatomical locations."""
    return {
        "locations": LOCATION_LABELS,
        "count": len(LOCATION_LABELS),
    }


@app.post("/demo/predict", response_model=AnalysisResponse)
async def demo_predict(modality: str = "CTA"):
    """
    Demo endpoint - returns predictions using simulation.
    """
    start_time = time.time()
    
    # Get inference engine (will use simulation if model not loaded)
    engine = get_inference_engine()
    result = engine._simulate_prediction()
    
    # Create predictions list
    predictions = []
    for location in LOCATION_LABELS:
        prob = result["probabilities"].get(location, 0.1)
        predictions.append(PredictionResult(
            location=location,
            probability=float(prob),
            detected=bool(prob > 0.5),
        ))
    
    processing_time = time.time() - start_time
    
    return AnalysisResponse(
        id="demo-" + str(uuid.uuid4())[:4],
        status="completed",
        predictions=predictions,
        overall_risk=result["risk_level"],
        confidence=float(result["max_probability"]),
        processing_time=processing_time,
        model_loaded=result.get("model_loaded", False),
        image_base64=result.get("image_base64"),
        slice_index=result.get("slice_index"),
        bbox=result.get("bbox"),
        modality=result.get("modality"),
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
