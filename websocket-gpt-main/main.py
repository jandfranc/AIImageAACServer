# main.py

import base64
from pathlib import Path
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
import json
from serverfunctions import (
    autocorrect,
    generate_images,
    log_event,
    predict,
    save_transcription_to_db,
    speech_recognition,
    synthesise,
    upload_audio,
)  # Ensure this import is correct
import logging
import sys
import os
from datetime import datetime, timedelta, timezone
import uuid

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Set up logging to print to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

logger.info("Server is running...")

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    logger.info("Sending connection response")
    emit('response', {"message": "Connected to server"})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

# Define separate event handlers for each request type
@socketio.on('PREDICT')
def handle_predict(message):
    logger.info(f"Received PREDICT message: {message}")
    try:
        response = predict("PREDICT", message.get("data", {}))
        logger.info(f"Sending PREDICT response: {response}")
        emit('PREDICT_response', response)
    except Exception as e:
        logger.error(f"Error handling PREDICT: {e}")
        emit('PREDICT_response', {"error": str(e)})

@socketio.on('SYNTHESISE')
def handle_synthesise(message):
    logger.info(f"Received SYNTHESISE message: {message}")
    try:
        response = synthesise("SYNTHESISE", message.get("data", {}))
        logger.info(f"Sending SYNTHESISE response: {response}")
        emit('SYNTHESISE_response', response)
    except Exception as e:
        logger.error(f"Error handling SYNTHESISE: {e}")
        emit('SYNTHESISE_response', {"error": str(e)})

@socketio.on('AUTOCOMPLETE')
def handle_autocomplete(message):
    logger.info(f"Received AUTOCOMPLETE message: {message}")
    try:
        response = autocorrect("AUTOCOMPLETE", message.get("data", {}))
        logger.info(f"Sending AUTOCOMPLETE response: {response}")
        emit('AUTOCOMPLETE_response', response)
    except Exception as e:
        logger.error(f"Error handling AUTOCOMPLETE: {e}")
        emit('AUTOCOMPLETE_response', {"error": str(e)})

@socketio.on('AUDIO_DATA')
def handle_audio_data(message):
    logger.info(f"Received AUDIO_DATA message: {message}")
    try:
        response = speech_recognition("AUDIO_DATA", message.get("data", {}))
        logger.info(f"Sending AUDIO_DATA response: {response}")
        emit('AUDIO_DATA_response', response)
    except Exception as e:
        logger.error(f"Error handling AUDIO_DATA: {e}")
        emit('AUDIO_DATA_response', {"error": str(e)})

@socketio.on('EVENT')
def handle_event(message):
    logger.info(f"Received EVENT message: {message}")
    try:
        response = log_event("EVENT", message.get("data", {}))
        logger.info(f"Sending EVENT response: {response}")
        emit('EVENT_response', response)
    except Exception as e:
        logger.error(f"Error handling EVENT: {e}")
        emit('EVENT_response', {"error": str(e)})

@socketio.on('GENERATE_IMAGE')
def handle_generate_image(message):
    logger.info(f"Received GENERATE_IMAGE message: {message}")
    try:
        response = generate_images("GENERATE_IMAGE", message)
        logger.info(f"Sending GENERATE_IMAGE response: {response}")
        emit('GENERATE_IMAGE_response', response)
    except Exception as e:
        logger.error(f"Error handling GENERATE_IMAGE: {e}")
        emit('GENERATE_IMAGE_response', {"error": str(e)})

@socketio.on('UPLOAD_AUDIO')
def handle_upload_audio(message):
    logger.info(f"Received UPLOAD_AUDIO message: {message}")
    try:
        response = upload_audio("UPLOAD_AUDIO", message)
        logger.info(f"Sending UPLOAD_AUDIO response: {response}")
        emit('UPLOAD_AUDIO_response', response)
        print(response)
    except Exception as e:
        logger.error(f"Error handling UPLOAD_AUDIO: {e}")
        emit('UPLOAD_AUDIO_response', {"error": str(e)})
        print(e)

@app.route('/socket.io/')
def socket_io_handler():
    return "Socket.IO is running"

@app.route('/audio/<filename>')
def serve_audio(filename):
    print(filename)
    return send_from_directory('audio_files', filename)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    logger.warning(f"Unexpected HTTP request: {request.method} {request.path}")
    return "404 Not Found", 404

if __name__ == '__main__':
    ip = "0.0.0.0"  # Bind to all IPv4 addresses
    port = 8000
    logger.info(f"Websocket hosted on: ws://{ip}:{port}")

    socketio.run(app, host=ip, port=port, debug=True)
