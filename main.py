import uuid
from fastapi import FastAPI, UploadFile, Form, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from datetime import datetime
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import random
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Add CORS middleware


# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, or specify explicitly, e.g., ["GET", "POST"]
    allow_headers=["*"],  # Allow all headers, or specify explicitly
)

# Ensure directories exist
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
AUDIO_DIR = BASE_DIR / "audio"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Serve static files
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


# Schema for text input
class GenerateImageRequest(BaseModel):
    text: str
    token: str


# Token validation
EXPECTED_TOKEN = "expected-token"


# Generate and save images
def generate_and_save_images(text: str):
    # Placeholder: Generate a simple text-based "image"


    image_paths = []
    colors = [tuple(random.randint(0, 255) for _ in range(3)) for i in range(4)]

    for i in range(4):
        # gen random color plssss
        color = colors[i]
        img = Image.new("RGB", (200, 200), color = color)
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), text, fill=(0, 0, 0))

        filename = f"image-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}.png"
        filepath = IMAGES_DIR / filename
        img.save(filepath)
        image_paths.append(f"http://130.237.67.212:8000/images/{filename}")
    return image_paths


# POST route to generate images
@app.post("/generate-images")
async def generate_images(request: GenerateImageRequest):
    if request.token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    image_urls = generate_and_save_images(request.text)
    return {"imageUrls": image_urls}


# POST route to upload audio
@app.post("/upload-audio")
async def upload_audio(
    audio: UploadFile, token: str = Header(None)
):
    if not token or token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Save the audio file
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{audio.filename}"
    file_path = AUDIO_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    audio_url = f"http://130.237.67.212:8000/audio/{filename}"
    return {"audioUrl": audio_url}

# POST route for TTS
@app.post("/tts")
async def text_to_speech(
    audio_url: str = Form(...),  # URL to the existing audio file on the server
    text: str = Form(...),  # Text to be synthesized
    token: str = Header(None)  # Authorization token
):
    if not token or token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Validate the audio file URL
    audio_filename = audio_url.split("/")[-1]
    source_audio_path = AUDIO_DIR / audio_filename
    if not source_audio_path.exists():
        raise HTTPException(status_code=404, detail="Source audio file not found")

    # Generate the TTS output file
    output_filename = f"tts-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}.wav"
    output_file_path = AUDIO_DIR / output_filename

    def format_sentence(sentence):
        if not sentence:
            return ""
        return sentence[0].upper() + sentence[1:].lower()

    try:
        tts.tts_to_file(
            text=format_sentence(text),
            speaker_wav=str(source_audio_path),
            language="en",  
            file_path=str(output_file_path),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    # Return the generated audio file URL
    output_audio_url = f"http://130.237.67.212:8000/audio/{output_filename}"
    return {"audioUrl": output_audio_url}


# Debug mode: print received requests (optional)
@app.middleware("http")
async def print_request(request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response


# Run the server with: uvicorn main:app --reload
