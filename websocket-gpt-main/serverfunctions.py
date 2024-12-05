# serverfunctions.py

from pathlib import Path
import random
import shutil
from PIL import Image, ImageDraw
import uuid
from fastapi import HTTPException
import requests
import json

import torch
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from pydub import AudioSegment
import io
import sqlite3
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import logging
from TTS.api import TTS
import sys
from dotenv import load_dotenv

load_dotenv()

# Initialize logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

tts = ""

# Initialize TTS
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
AUDIO_DIR = BASE_DIR / "audio_files"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Load environment variables from .env file
load_dotenv(".env")

# Read the system prompt from a file
with open("SystemPrompt.txt", "r") as file:
    system_prompt = file.read()

# Load config
with open('config/config.json') as config_file:
    config = json.load(config_file)

def setup_database():
    """
    Set up the SQLite database and create necessary tables if they don't exist.
    """
    conn = sqlite3.connect("transcriptions.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            audio_link TEXT NOT NULL,
            is_synthesized BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

setup_database()

def save_transcription_to_db(text, audio_link, is_synthesized):
    """
    Save a transcription to the database.
    """
    try:
        conn = sqlite3.connect("transcriptions.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO transcriptions (text, audio_link, is_synthesized, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
            (text, audio_link, is_synthesized, datetime.now(timezone.utc), datetime.now(timezone.utc)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Error saving to database: %s", e)

def save_transcription_segment_to_db(text, start_offset, end_offset, request_time):
    """
    Save a transcription segment to the database.
    """
    try:
        start_time = request_time + timedelta(seconds=start_offset)
        end_time = request_time + timedelta(seconds=end_offset)
        audio_url = ""

        conn = sqlite3.connect("transcriptions.db")
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO transcriptions (text, audio_link, is_synthesized, start_time, end_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            (text, audio_url, False, start_time, end_time),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Error saving to database: %s", e)

def transcribe_audio(file_path, request_time):
    """
    Transcribe audio file using a transcription server.
    """
    try:
        url = config["transcriptionServer"]
        files = {"file": open(file_path, "rb")}
        response = requests.post(url, files=files)
        logger.info(response.json())

        if response.status_code == 200:
            transcription_result = response.json()
            logger.info("Transcription result:")
            logger.info(json.dumps(transcription_result, indent=4))

            if (
                "segments" in transcription_result
                and len(transcription_result["segments"]) > 0
            ):
                segments = transcription_result["segments"]
                for segment in segments:
                    text = segment.get("text", "")
                    start_time = segment.get("start", 0.0)
                    end_time = segment.get("end", 0.0)
                    
                    if text:
                        save_transcription_segment_to_db(text, start_time, end_time, request_time)

                transcription_text = segments[0].get("text", "")
                return transcription_text if transcription_text else "No audible voice."
            else:
                return "No audible voice."
        else:
            logger.error("Error: %s", json.dumps(response.json(), indent=4))
            return f"Error transcribing audio. {response}"
    except Exception as e:
        logger.error("Error during transcription: %s", e)
        return "Error transcribing audio."

def get_last_ten_minutes_transcriptions():
    """
    Retrieve transcriptions from the last ten minutes from the database.
    """
    try:
        conn = sqlite3.connect("transcriptions.db")
        c = conn.cursor()

        now_utc = datetime.now(timezone.utc)
        ten_minutes_ago_utc = now_utc - timedelta(minutes=10)
        ten_minutes_ago_str = ten_minutes_ago_utc.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Retrieving transcriptions since: {ten_minutes_ago_str}")

        c.execute(
            """
            SELECT text, is_synthesized FROM transcriptions
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """,
            (ten_minutes_ago_str,),
        )

        rows = c.fetchall()
        conn.close()

        transcriptions = []
        for row in rows:
            prefix = "user:" if row[1] else "other:"
            transcriptions.append(f"{prefix} {row[0]}")

        transcription_string = "\n".join(transcriptions)

        if len(transcription_string) > 4000:
            while len(transcription_string) > 4000:
                first_newline_pos = transcription_string.find("\n")
                if first_newline_pos == -1:
                    break
                transcription_string = transcription_string[first_newline_pos + 1 :]
        return transcription_string

    except Exception as e:
        logger.error("Error retrieving transcriptions: %s", e)
        return "Error retrieving transcriptions."

def log_event(request_type, event_data):
    """
    Log an event to the database.
    """
    try:
        conn = sqlite3.connect("transcriptions.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO events (event_type, event_data, timestamp) VALUES (?, ?, ?)",
            (request_type, json.dumps(event_data), datetime.now(timezone.utc)),
        )
        conn.commit()
        conn.close()
        return {"message": "Event logged successfully."}
    except Exception as e:
        logger.error("Error saving event to database: %s", e)
        return {"error": "Failed to log event."}

def process_audio_data(audio_data):
    """
    Process audio data and save it as a WAV file.
    """
    try:
        missing_padding = len(audio_data) % 4
        if missing_padding != 0:
            audio_data += "=" * (4 - missing_padding)
        
        
        decoded_audio = base64.b64decode(audio_data)

        audio_segment = AudioSegment.from_file(io.BytesIO(decoded_audio), format="webm")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"audio_{timestamp}_{unique_id}.wav"

        output_directory = "audio_files"
        os.makedirs(output_directory, exist_ok=True)

        output_path = os.path.join(output_directory, output_filename)
        audio_segment.export(output_path, format="wav")

        logger.info("Audio data processed and saved as WAV to: %s", output_path)
        return output_path

    except Exception as e:
        logger.error("Error processing audio data: %s", e)
        return None

def predict(request_type, data):
    """
    Generate a prediction using OpenAI GPT-4 model.
    """
    try:
        text = data["text"]
        text += get_last_ten_minutes_transcriptions()
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            n=3,
            temperature=1,
        )

        response = {
            "request_type": request_type,
            "data": {
                "options": [
                    response.choices[0].message.content,
                    response.choices[1].message.content,
                    response.choices[2].message.content,
                ]
            },
        }

        return response
    except Exception as e:
        logger.error(f"Error in predict function: {e}")
        return {"error": "Prediction failed"}

def synthesise(request_type, data):
    """
    Synthesize text into speech using a TTS server.
    """
    try:
        print(data)
        text = data["text"]
        audio_url = data["audio_file"]  

        if not audio_url:
            raise ValueError("audio_url is required for synthesise")


        audio_filename = audio_url.split("/")[-1]
        source_audio_path = AUDIO_DIR / audio_filename
        print(source_audio_path)

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
        generated_audio_url = f"http://130.237.67.212:8000/audio/{output_filename}"

        save_transcription_to_db(text, generated_audio_url, True)

        response = {"request_type": request_type, "data": {"audio_url": generated_audio_url}}

        return response
    except Exception as e:
        logger.error(f"Error in synthesise function: {e}")
        return {"error": "Synthesis failed"}

def upload_audio(request_type, data):
    """
    Upload an audio file to the server.
    """
    data = data["data"]
    try:
        audio = data["audio"]
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{data['filename']}"
        file_path = AUDIO_DIR / filename

        # Decode the base64 audio data
        audio_path = process_audio_data(audio)
        if not audio_path:
            raise ValueError("Audio processing failed")
        print(audio_path)
        audio_url = f"http://130.237.67.212:8000/audio/{audio_path.split('/')[-1]}"
        print(audio_url)
        return {"request_type": request_type, "data": {"audio_url": audio_url}}
    except Exception as e:
        logger.error(f"Error in upload_audio function: {e}")
        return {"error": "Audio upload failed"}

def autocorrect(request_type, data):
    """
    Perform autocorrection on input text.
    """
    try:
        text = data["input"]
        # Placeholder for actual autocorrect logic
        # For demonstration, simply append some words
        response = {
            "request_type": request_type,
            "data": {"options": [text + " word1", text + " word2", text + " word3"]},
        }
        return response
    except Exception as e:
        logger.error(f"Error in autocorrect function: {e}")
        return {"error": "Autocorrect failed"}

def speech_recognition(request_type, data):
    """
    Perform speech recognition on audio data.
    """
    try:
        audio_data = data["audio"]
        audio_path = process_audio_data(audio_data)

        if audio_path is None:
            return {
                "request_type": request_type,
                "data": "No speech detected in the audio segment.",
            }

        request_time = datetime.now(timezone.utc)
        transcription = transcribe_audio(audio_path, request_time)
        audio_url = f"http://130.237.67.212:8000/audio/{Path(audio_path).name}"

        if transcription != "Error transcribing audio.":
            save_transcription_to_db(transcription, audio_url, False)

        response = {"request_type": request_type, "data": transcription, "language": "en"}

        return response
    except Exception as e:
        logger.error(f"Error in speech_recognition function: {e}")
        return {"error": "Speech recognition failed"}

def generate_and_save_images(text: str):
    """
    Generate simple text-based images with random background colors.
    """
    try:
        image_paths = []
        colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(4)]

        for i in range(4):
            color = colors[i]
            img = Image.new("RGB", (200, 200), color=color)
            draw = ImageDraw.Draw(img)
            text_to_draw = text[:15]  # Limit text length for the image
            draw.text((50, 90), text_to_draw, fill=(0, 0, 0))

            filename = f"image-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}.png"
            filepath = IMAGES_DIR / filename
            img.save(filepath)
            image_paths.append(f"http://130.237.67.212:8000/images/{filename}")
        return image_paths
    except Exception as e:
        logger.error(f"Error in generate_and_save_images function: {e}")
        return []

def generate_images(request_type, data):
    """
    Generate images based on input text.
    """
    try:

        print(data)
        text = data["text"]
        logger.info(f"Generating images for text: {text}")
        image_urls = generate_and_save_images(text)
        if not image_urls:
            return {"error": "Image generation failed"}
        return {"request_type": request_type, "data": {"imageUrls": image_urls}}
    except Exception as e:
        logger.error(f"Error in generate_images function: {e}")
        return {"error": "Image generation failed"}
