from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import shutil
import os
import cv2
import numpy as np
import tensorflow as tf
import gdown  # <--- NEW IMPORT

# 1. USE LEGACY KERAS
import tf_keras as keras
from tf_keras import layers, models, applications
from mtcnn import MTCNN

# CONFIGURATION
MODEL_PATH = "deepfake_model.h5"
IMG_SIZE = 224
SEQ_LENGTH = 10 

DRIVE_FILE_ID = "1OLL19HAOMOwTjwVzU-TefC7ECz3msm4P" 

app = FastAPI()

# --- NEW: DOWNLOADER FUNCTION ---
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found. Downloading from Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            print("Download complete!")
        except Exception as e:
            print(f"Failed to download model: {e}")

# 2. BUILD MODEL & LOAD WEIGHTS
def build_pro_model():
    base_model = applications.ResNet50V2(include_top=False, weights=None, pooling='avg')
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    inputs = layers.Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    encoded = layers.TimeDistributed(base_model)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(encoded)
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs, outputs)

print("Loading Model...")
try:
    download_model_if_missing()
    
    model = build_pro_model()
    model.load_weights(MODEL_PATH)
    print("Model Loaded!")
except Exception as e:
    print(f"Error: {e}")
    model = None

detector = MTCNN()

# 3. HELPER: EXTRACT FACES
def extract_faces(video_path, target_frame_count):
    if not os.path.exists(video_path): return None

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if not frames: return None

    # Calculate stride (skip) to get exactly the number of frames user asked for
    total_frames = len(frames)
    stride = max(1, total_frames // target_frame_count)
    
    face_crops = []
    
    for i in range(0, total_frames, stride):
        if len(face_crops) >= target_frame_count: break
        
        frame = frames[i]
        results = detector.detect_faces(frame)
        
        if results:
            x, y, w, h = results[0]['box']
            # Ensure box is within frame
            x, y = max(0, x), max(0, y)
            face = frame[y:y+h, x:x+w]
            
            try:
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
                face_crops.append(face)
            except: 
                pass # Skip if resize fails

    if not face_crops: return None

    # Pad if we didn't find enough faces
    while len(face_crops) < target_frame_count:
        if len(face_crops) > 0:
            face_crops.append(face_crops[-1]) # Duplicate last face
        else:
            return None # No faces found at all
        
    return np.array(face_crops[:target_frame_count])

# 4. API ENDPOINT
@app.post("/predict")
async def predict_video(
    file: UploadFile = File(...), 
    frames_to_check: int = Form(20) 
):
    if model is None:
        return {"status": "error", "message": "Model failed to load."}

    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract Faces
        all_faces = extract_faces(temp_filename, frames_to_check)
        
        if all_faces is None:
            result = {"status": "error", "message": "No faces detected in video."}
        else:
            predictions = []
            
            # Batch Processing (chunk into groups of 10)
            for i in range(0, len(all_faces), SEQ_LENGTH):
                chunk = all_faces[i : i + SEQ_LENGTH]
                
                # Pad chunk if smaller than 10
                if len(chunk) < SEQ_LENGTH:
                    padding = [chunk[-1]] * (SEQ_LENGTH - len(chunk))
                    chunk = np.concatenate([chunk, padding])
                
                input_batch = np.expand_dims(chunk, axis=0)
                pred = model.predict(input_batch, verbose=0)[0][0]
                predictions.append(pred)
            
            avg_score = np.mean(predictions)
            
            if avg_score >= 0.5:
                label = "FAKE"
                confidence = float(avg_score * 100)
            else:
                label = "REAL"
                confidence = float((1 - avg_score) * 100)
                
            result = {
                "status": "success",
                "label": label,
                "confidence": f"{confidence:.2f}%",
                "frames_analyzed": len(all_faces),
                "raw_score": float(avg_score)
            }

    except Exception as e:
        result = {"status": "error", "message": str(e)}

    # Cleanup
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return result