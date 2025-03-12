
FROM python:3.10-slim

# Install dependencies
RUN apt-get update &&     apt-get install -y     libgl1-mesa-glx     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install fastapi uvicorn pyngrok nest_asyncio face_recognition dlib pillow numpy

# Copy project files
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
