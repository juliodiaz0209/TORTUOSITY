from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import base64
import io
from PIL import Image
import json
from typing import Optional, Dict, Any
import numpy as np
from skimage import exposure, img_as_ubyte, img_as_float32

# Import the tortuosity analysis functions
from Tortuosity import (
    load_maskrcnn_model,
    load_unet_model,
    predict_maskrcnn_model,
    predict_unet_model,
    show_combined_result,
    show_combined_result_with_models,
    resize_to_previous_multiple_of_32,
    device
)

# Create FastAPI app
app = FastAPI(
    title="Análisis de Tortuosidad Avanzado API",
    description="API para análisis de tortuosidad de glándulas de Meibomio usando PyTorch (Mask R-CNN & UNet)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for temporary files and results
TEMP_DIR = Path("temp")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model paths - with fallbacks
MASK_RCNN_MODEL_PATH = "final_model (11).pth"
UNET_MODEL_PATH = "final_model_tarsus_improved.pth"

# Fallback model paths
FALLBACK_MASK_RCNN_PATHS = [
    "final_model (11).pth",
    "final_model_tarsus.pth",
    "final_model.pth"
]

FALLBACK_UNET_PATHS = [
    "final_model_tarsus_improved.pth",
    "final_model_tarsus.pth"
]

# Global model instances (loaded once at startup)
maskrcnn_model = None
unet_model = None

def try_load_model_with_fallbacks(load_function, model_paths, model_name):
    """Try to load a model from multiple possible paths"""
    for path in model_paths:
        try:
            print(f"Trying to load {model_name} from: {path}")
            if os.path.exists(path):
                model = load_function(path)
                print(f"Successfully loaded {model_name} from: {path}")
                return model
            else:
                print(f"Model file not found: {path}")
        except Exception as e:
            print(f"Failed to load {model_name} from {path}: {e}")
            continue
    
    raise Exception(f"Could not load {model_name} from any of the provided paths: {model_paths}")

def clahe_like_imagej(img, block_radius=63, bins=255, slope=3.0, convert_to_gray=True):
    """
    Replica el CLAHE de ImageJ con un error ≤ 1 nivel de gris.
    img : uint8 ó RGB uint8
    convert_to_gray: Si True, convierte RGB a gris antes de aplicar CLAHE
    """
    tile        = 2*block_radius + 1
    clip_limit  = slope / bins      # mapeo exacto
    nbins       = bins + 1

    if img.ndim == 3 and convert_to_gray:
        # Convertir RGB a gris usando la fórmula estándar
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        out = exposure.equalize_adapthist(gray_img,
                                          kernel_size=tile,
                                          nbins=nbins,
                                          clip_limit=clip_limit)
        return img_as_ubyte(out)

    elif img.ndim == 2:                        # ya es gris
        out = exposure.equalize_adapthist(img,
                                          kernel_size=tile,
                                          nbins=nbins,
                                          clip_limit=clip_limit)
        return img_as_ubyte(out)

    elif img.ndim == 3 and not convert_to_gray:  # color: trata cada canal
        ch = [clahe_like_imagej(img[..., c],
                                block_radius, bins, slope, convert_to_gray=False)
              for c in range(img.shape[2])]
        return np.stack(ch, axis=-1)

    else:
        raise ValueError("Solo imágenes 2D o RGB.")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global maskrcnn_model, unet_model
    try:
        print("Starting model loading process...")
        print(f"Mask R-CNN model path: {MASK_RCNN_MODEL_PATH}")
        print(f"UNet model path: {UNET_MODEL_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        
        # Check if model files exist
        for path in FALLBACK_MASK_RCNN_PATHS + FALLBACK_UNET_PATHS:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"✓ Model file exists: {path} ({size:,} bytes)")
                # Try to read first few bytes to check if file is readable
                try:
                    with open(path, 'rb') as f:
                        header = f.read(16)
                        print(f"  File header (hex): {header.hex()}")
                except Exception as e:
                    print(f"  Warning: Could not read file header: {e}")
            else:
                print(f"✗ Model file missing: {path}")
        
        # Print system information
        import platform
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"Architecture: {platform.architecture()}")
        
        # Print PyTorch information
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
        
        # Print memory information
        import psutil
        try:
            memory = psutil.virtual_memory()
            print(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        except ImportError:
            print("psutil not available for memory info")
        
        # Add a small delay to ensure system is ready
        import asyncio
        await asyncio.sleep(3)
        
        print("Loading Mask R-CNN model...")
        try:
            maskrcnn_model = try_load_model_with_fallbacks(lambda path: load_maskrcnn_model(path, device), FALLBACK_MASK_RCNN_PATHS, "Mask R-CNN")
            print("✓ Mask R-CNN model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Mask R-CNN model: {e}")
            maskrcnn_model = None
        
        print("Loading UNet model...")
        try:
            unet_model = try_load_model_with_fallbacks(lambda path: load_unet_model(path, device), FALLBACK_UNET_PATHS, "UNet")
            print("✓ UNet model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load UNet model: {e}")
            unet_model = None
        
        if maskrcnn_model is not None and unet_model is not None:
            print("✓ All models loaded successfully!")
        else:
            print("⚠ Some models failed to load, but continuing startup...")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise the error immediately, let the app start and handle it gracefully
        print("Warning: Models failed to load, but continuing startup...")
        maskrcnn_model = None
        unet_model = None
    
    # Final status check
    if maskrcnn_model is not None and unet_model is not None:
        print("🎉 Application startup completed successfully with all models loaded!")
    else:
        print("⚠ Application startup completed with degraded functionality - some models failed to load")
        print("   The application will continue to run but may not be able to process images")

@app.get("/")
async def root():
    """Serve the main HTML interface"""
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Análisis de Tortuosidad Avanzado API",
        "version": "1.0.0",
        "endpoints": {
            "/": "Main interface",
            "/api": "API information",
            "/health": "Health check",
            "/models/status": "Detailed model status",
            "/analyze": "Analyze image for tortuosity",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = maskrcnn_model is not None and unet_model is not None
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": {
            "maskrcnn": maskrcnn_model is not None,
            "unet": unet_model is not None
        },
        "device": str(device),
        "message": "Models loaded successfully" if models_loaded else "Some models failed to load"
    }

@app.get("/models/status")
async def model_status():
    """Detailed model status endpoint"""
    return {
        "models": {
            "maskrcnn": {
                "status": "loaded" if maskrcnn_model is not None else "failed",
                "path": MASK_RCNN_MODEL_PATH,
                "fallback_paths": FALLBACK_MASK_RCNN_PATHS
            },
            "unet": {
                "status": "loaded" if unet_model is not None else "failed",
                "path": UNET_MODEL_PATH,
                "fallback_paths": FALLBACK_UNET_PATHS
            }
        },
        "device": str(device),
        "working_directory": os.getcwd(),
        "available_files": os.listdir('.'),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze an uploaded image for gland tortuosity
    
    Args:
        file: Image file (jpg, jpeg, png)
        
    Returns:
        JSON with analysis results including:
        - processed_image: Base64 encoded processed image
        - avg_tortuosity: Average tortuosity value
        - num_glands: Number of detected glands
        - individual_tortuosities: List of individual gland tortuosities
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File extension {file_extension} not allowed. Use: {allowed_extensions}"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Check if models are loaded
        if maskrcnn_model is None or unet_model is None:
            model_status = {
                "maskrcnn": "loaded" if maskrcnn_model is not None else "failed",
                "unet": "loaded" if unet_model is not None else "failed"
            }
            raise HTTPException(
                status_code=503, 
                detail={
                    "error": "Service temporarily unavailable",
                    "message": "AI models are not currently loaded. Please try again later.",
                    "model_status": model_status,
                    "suggestion": "Contact administrator if the problem persists."
                }
            )
        
        # Perform analysis using pre-loaded models
        result_image, tortuosity_data = show_combined_result_with_models(
            temp_file_path, 
            maskrcnn_model, 
            unet_model, 
            device
        )
        
        # Convert result image to base64
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Clean up temporary file
        background_tasks.add_task(os.unlink, temp_file_path) if background_tasks else os.unlink(temp_file_path)
        
        # Return results
        return {
            "success": True,
            "message": "Analysis completed successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "avg_tortuosity": round(tortuosity_data['avg_tortuosity'], 3),
                "num_glands": tortuosity_data['num_glands'],
                "individual_tortuosities": [round(t, 3) for t in tortuosity_data['individual_tortuosities']],
                "analysis_info": {
                    "total_glands_analyzed": len(tortuosity_data['individual_tortuosities']),
                    "tortuosity_range": {
                        "min": round(min(tortuosity_data['individual_tortuosities']), 3) if tortuosity_data['individual_tortuosities'] else 0,
                        "max": round(max(tortuosity_data['individual_tortuosities']), 3) if tortuosity_data['individual_tortuosities'] else 0
                    }
                }
            }
        }
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model file not found: {str(e)}"
        )
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/apply-clahe")
async def apply_clahe_filter(
    file: UploadFile = File(...),
    block_radius: int = 63,
    bins: int = 255,
    slope: float = 3.0,
    convert_to_gray: bool = True
):
    """
    Apply CLAHE filter to an uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
        block_radius: Block radius for CLAHE (default: 63)
        bins: Number of bins for histogram (default: 255)
        slope: Slope for clip limit (default: 3.0)
        
    Returns:
        JSON with processed image as base64
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File extension {file_extension} not allowed. Use: {allowed_extensions}"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Load image and convert to numpy array
        image = Image.open(temp_file_path).convert("RGB")
        img_array = np.array(image)
        
        # Apply CLAHE filter
        clahe_img = clahe_like_imagej(img_array, block_radius, bins, slope, convert_to_gray)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(clahe_img)
        
        # Convert result image to base64
        img_buffer = io.BytesIO()
        processed_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Return results
        return {
            "success": True,
            "message": "CLAHE filter applied successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "parameters": {
                    "block_radius": block_radius,
                    "bins": bins,
                    "slope": slope,
                    "convert_to_gray": convert_to_gray
                }
            }
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"CLAHE processing failed: {str(e)}"
        )

@app.get("/info")
async def get_analysis_info():
    """Get information about the tortuosity analysis"""
    return {
        "description": "Análisis de Tortuosidad de Glándulas de Meibomio",
        "methodology": {
            "tortuosity_formula": "Tortuosidad = (Perímetro / (2 × Altura del rectángulo mínimo externo)) - 1",
            "interpretation": {
                "low": "0.0 - 0.1: Tortuosidad baja (generalmente normal)",
                "moderate": "0.1 - 0.2: Tortuosidad moderada (puede indicar cambios iniciales)",
                "high": "> 0.2: Tortuosidad alta (sugestivo de MGD, requiere correlación clínica)"
            }
        },
        "models_used": {
            "mask_rcnn": "Detección y segmentación de glándulas individuales",
            "unet": "Segmentación del contorno del párpado (Tarsus)"
        },
        "note": "Los rangos de interpretación son aproximados y la interpretación final debe ser realizada por un especialista."
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 