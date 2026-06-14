from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import shutil
import os
import uuid
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
yolo_model = YOLO("best.pt")
audio_model = load_model("Model1.h5")

class_names = ["unripe", "ripe", "overripe"]


def predict_image(image_path):
    results = yolo_model(image_path, conf=0.1)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return "No detection", 0.0, []

    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = yolo_model.names[cls_id]

        detections.append({
            "class": class_name,
            "confidence": round(conf, 2),
            "box": box.xyxy[0].tolist()
        })

    best_detection = detections[0]

    return (
        best_detection["class"],
        best_detection["confidence"],
        detections
    )


def predict_audio_file(audio_path):
    wav_path = f"{uuid.uuid4()}.wav"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", audio_path,
        wav_path
    ], check=True)

    audio_data, sample_rate = librosa.load(wav_path, res_type="kaiser_fast")

    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=128
    )

    feature = np.mean(mfcc.T, axis=0)
    feature = np.array([feature])

    prediction = audio_model.predict(feature)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label = class_names[predicted_index]

    if os.path.exists(wav_path):
        os.remove(wav_path)

    return label, confidence


def fuse_results(image_label, image_conf, audio_label, audio_conf):
    image_weight = 0.6
    audio_weight = 0.4

    scores = {
        "unripe": 0.0,
        "ripe": 0.0,
        "overripe": 0.0
    }

    if image_label in scores:
        scores[image_label] += image_conf * image_weight

    if audio_label in scores:
        scores[audio_label] += audio_conf * audio_weight

    final_label = max(scores, key=scores.get)
    final_confidence = scores[final_label]

    return final_label, final_confidence, scores


@app.get("/")
def home():
    return {
        "message": "Watermelon Ripeness API running",
        "yolo_classes": yolo_model.names,
        "audio_classes": class_names
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_path = f"{uuid.uuid4()}.jpg"

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image_label, image_conf, detections = predict_image(image_path)

        return {
            "class": image_label,
            "confidence": round(image_conf, 2),
            "detections": detections
        }

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    audio_path = f"{uuid.uuid4()}_{file.filename}"

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_label, audio_conf = predict_audio_file(audio_path)

        return {
            "class": audio_label,
            "confidence": round(audio_conf, 2)
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/predict-fusion")
async def predict_fusion(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    image_path = f"{uuid.uuid4()}.jpg"
    audio_path = f"{uuid.uuid4()}_{audio.filename}"

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        image_label, image_conf, detections = predict_image(image_path)
        audio_label, audio_conf = predict_audio_file(audio_path)

        final_label, final_confidence, fusion_scores = fuse_results(
            image_label,
            image_conf,
            audio_label,
            audio_conf
        )

        return {
            "image_result": {
                "class": image_label,
                "confidence": round(image_conf, 2),
                "detections": detections
            },
            "audio_result": {
                "class": audio_label,
                "confidence": round(audio_conf, 2)
            },
            "fusion_result": {
                "class": final_label,
                "confidence": round(final_confidence, 2),
                "scores": fusion_scores
            }
        }

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/predict-video-fusion")
async def predict_video_fusion(video: UploadFile = File(...)):
    video_path = f"{uuid.uuid4()}_{video.filename}"
    image_path = f"{uuid.uuid4()}.jpg"
    audio_path = f"{uuid.uuid4()}.wav"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", "00:00:01",
            "-vframes", "1",
            image_path
        ], check=True)

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            audio_path
        ], check=True)

        image_label, image_conf, detections = predict_image(image_path)
        audio_label, audio_conf = predict_audio_file(audio_path)

        final_label, final_confidence, fusion_scores = fuse_results(
            image_label,
            image_conf,
            audio_label,
            audio_conf
        )

        return {
            "image_result": {
                "class": image_label,
                "confidence": round(image_conf, 2),
                "detections": detections
            },
            "audio_result": {
                "class": audio_label,
                "confidence": round(audio_conf, 2)
            },
            "fusion_result": {
                "class": final_label,
                "confidence": round(final_confidence, 2),
                "scores": fusion_scores
            }
        }

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)