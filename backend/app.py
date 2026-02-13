from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import os
from dimensionality_reduction import reduce_to_3d_umap

app = FastAPI(title="Medical Embedding Visualization API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load cached data on startup
CACHE_DIR = './cache'
DATA_INFO = None
EMBEDDINGS = {}
TSNE_COORDS = {}


@app.on_event("startup")
async def load_cache():
    """Load all cached data when server starts"""
    global DATA_INFO, EMBEDDINGS, TSNE_COORDS

    print("Loading cached data...")

    # Load data info
    with open(f'{CACHE_DIR}/data_info.pkl', 'rb') as f:
        DATA_INFO = pickle.load(f)
    print(f"✓ Loaded {len(DATA_INFO['texts'])} samples")

    # Load embeddings (including triplet-biobert)
    for model_name in ['sentence-bert', 'biobert', 'clinical-bert', 'triplet-biobert']:
        try:
            EMBEDDINGS[model_name] = np.load(f'{CACHE_DIR}/embeddings_{model_name}.npy')
            print(f"✓ Loaded {model_name} embeddings: {EMBEDDINGS[model_name].shape}")
        except FileNotFoundError:
            print(f"⚠ Skipping {model_name} (not found)")

    # Load t-SNE coords (including triplet-biobert)
    for model_name in ['sentence-bert', 'biobert', 'clinical-bert', 'triplet-biobert']:
        try:
            TSNE_COORDS[model_name] = np.load(f'{CACHE_DIR}/tsne_{model_name}.npy')
            print(f"✓ Loaded {model_name} t-SNE coords: {TSNE_COORDS[model_name].shape}")
        except FileNotFoundError:
            print(f"⚠ Skipping {model_name} t-SNE (not found)")

    print("✓ Server ready!")


# API ENDPOINTS FIRST (before static mounts)

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "message": "Medical Embedding Visualization API"}


@app.get("/api/models")
async def get_models():
    """Get list of available models"""
    return {
        "models": [
            {"id": "sentence-bert", "name": "Sentence-BERT", "domain": "general"},
            {"id": "biobert", "name": "BioBERT", "domain": "biomedical"},
            {"id": "clinical-bert", "name": "ClinicalBERT", "domain": "clinical"},
            {"id": "triplet-biobert", "name": "Fine-tuned BioBERT (Triplet Loss)", "domain": "medical-finetuned"}
        ]
    }


@app.get("/api/specialties")
async def get_specialties():
    """Get list of available specialties"""
    unique_specialties = list(set(DATA_INFO['specialties']))
    specialty_counts = {s: DATA_INFO['specialties'].count(s) for s in unique_specialties}

    return {
        "specialties": [
            {"name": s, "count": specialty_counts[s]}
            for s in sorted(unique_specialties)
        ]
    }


@app.get("/api/visualize/{model_id}")
async def get_visualization(
        model_id: str,
        reduction: str = "tsne",
        specialties: str = None
):
    """Get 3D coordinates for visualization"""

    if model_id not in EMBEDDINGS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_id}' not found"}
        )

    # Get coordinates based on reduction method
    if reduction == "tsne":
        if model_id not in TSNE_COORDS:
            return JSONResponse(
                status_code=404,
                content={"error": f"t-SNE coordinates not pre-computed for '{model_id}'"}
            )
        coords_3d = TSNE_COORDS[model_id]
    elif reduction == "umap":
        print(f"Computing UMAP for {model_id}...")
        coords_3d = reduce_to_3d_umap(EMBEDDINGS[model_id])
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid reduction method: {reduction}"}
        )

    # Filter by specialties if provided
    if specialties:
        selected_specialties = [s.strip() for s in specialties.split(',')]
        indices = [i for i, s in enumerate(DATA_INFO['specialties']) if s in selected_specialties]
    else:
        indices = list(range(len(DATA_INFO['texts'])))

    # Prepare response data
    points = []
    for idx in indices:
        points.append({
            "id": idx,
            "x": float(coords_3d[idx][0]),
            "y": float(coords_3d[idx][1]),
            "z": float(coords_3d[idx][2]),
            "specialty": DATA_INFO['specialties'][idx],
            "text_snippet": DATA_INFO['texts'][idx][:200] + "..." if len(DATA_INFO['texts'][idx]) > 200 else
            DATA_INFO['texts'][idx],
            "full_text": DATA_INFO['texts'][idx]
        })

    return {
        "model": model_id,
        "reduction": reduction,
        "total_points": len(points),
        "points": points
    }


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics"""
    return {
        "total_samples": len(DATA_INFO['texts']),
        "specialties": len(set(DATA_INFO['specialties'])),
        "models": len(EMBEDDINGS),
        "embedding_dimensions": {
            model: embeddings.shape[1]
            for model, embeddings in EMBEDDINGS.items()
        }
    }


# ROOT ENDPOINT - serves index.html
@app.get("/")
async def serve_root():
    """Serve the main HTML page"""
    return FileResponse("../frontend/index.html")


# STATIC FILE MOUNTS (MUST be last!)
app.mount("/css", StaticFiles(directory="../frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="../frontend/js"), name="js")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)