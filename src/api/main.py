from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.inference import InferenceEngine
from src.utils.image_utils import preprocess_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engines = {
        "autoencoder": InferenceEngine("autoencoder"),
        "vit": InferenceEngine("vit"),
    }
    yield
    app.state.engines.clear()


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "industrial-defect-detection-api",
    }


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/models")
def models():
    return {"models": list(app.state.engines.keys())}


@app.post("/predict/{model_name}")
async def predict(
    model_name: str,
    file: UploadFile = File(...),
):
    if model_name not in app.state.engines:
        raise HTTPException(
            status_code=400,
            detail="Supported models: autoencoder, vit",
        )

    try:
        image = preprocess_image(await file.read())

        result = app.state.engines[model_name].predict(image)

        return {
            "model": model_name,
            **result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )