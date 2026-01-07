from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid

from video_inference import process_video, get_model

app = FastAPI()

# --- CONFIG ---
TEMP_DIR = "temp"
MODEL_PATH = "best_deepfake_pro.pth"

os.makedirs(TEMP_DIR, exist_ok=True)

# --- Load model ONCE (very important) ---
try:
    model = get_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model load failed: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---- Basic validation ----
    if not file.filename.endswith((".mp4", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # Unique temp filename (important for parallel requests)
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    video_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        # Save uploaded video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---- Inference ----
        label, confidence = process_video(video_path, model)

        return {
            "prediction": label,                     # 🔴 FAKE / 🟢 REAL / 🟡 UNCERTAIN
            "confidence": round(confidence * 100, 2) # percentage
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
