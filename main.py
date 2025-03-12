
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import json
import os
from PIL import Image
from io import BytesIO
import nest_asyncio
from pyngrok import ngrok

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENCODINGS_FILE = "/content/face_encodings.json"

def load_encodings_from_file(file_path=ENCODINGS_FILE):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        return []
    with open(file_path, "r") as file:
        return json.load(file)

def get_face_encoding(image_data):
    image = np.array(Image.open(BytesIO(image_data)))
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return None
    return encodings[0].tolist()

@app.post("/compare/")
async def compare_faces(image: UploadFile = File(...)):
    stored_encodings = load_encodings_from_file()

    if not stored_encodings:
        raise HTTPException(status_code=404, detail="No stored encodings found. Please add images first.")

    image_data = await image.read()
    img_encoding = get_face_encoding(image_data)
    if img_encoding is None:
        raise HTTPException(status_code=400, detail="No face found in the provided image.")

    known_encodings = [np.array(entry["encoding"]) for entry in stored_encodings]
    image_names = [entry["name"] for entry in stored_encodings]

    face_distances = face_recognition.face_distance(known_encodings, np.array(img_encoding))
    matches = list(zip(image_names, face_distances))
    matches.sort(key=lambda x: x[1])

    top_matches = [{"name": name, "similarity_score": f"{100 - (distance * 100):.2f}%"} for name, distance in matches[:20]]

    return {"top_matches": top_matches}

@app.post("/add/")
async def add_encoding(image: UploadFile = File(...)):
    stored_encodings = load_encodings_from_file()

    image_data = await image.read()
    img_encoding = get_face_encoding(image_data)
    if img_encoding is None:
        raise HTTPException(status_code=400, detail="No face found in the provided image.")

    stored_encodings.append({"name": image.filename, "encoding": img_encoding})
    with open(ENCODINGS_FILE, "w") as file:
        json.dump(stored_encodings, file)

    return {"message": f"Added encoding for {image.filename}"}

@app.get("/")
def read_root():
    return {"message": "Face Recognition API is running!"}

nest_asyncio.apply()
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)

uvicorn.run(app, host="0.0.0.0", port=8000)
