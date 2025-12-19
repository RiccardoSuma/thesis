import torch
import gc # Garbage Collector
from modules.storage.qdrant_ops import VectorDB
from modules.video.video_processing import VideoIngestor
from modules.audio.video_transcribe import videoTranscriber

# 1. Setup DB
db = VectorDB(collection_name="thesis_rag")

# 2. Paths
VIDEO_PATH = r"data\videos\LezioneDL_01_31_03.mp4" # Use raw string for Windows paths
VIDEO_NAME = "LezioneDL_01.mp4"
AUDIO_PATH = r"data\audios\LezioneDL_01_31_03_AUDIO.wav"



# --- PHASE 1: AUDIO (Whisper) ---
print("--- STARTING PHASE 1: AUDIO ---")

transcriber = videoTranscriber() # Load Whisper (Turbo)

# Extract
transcriber.extract_audio_from_video(video_path=VIDEO_PATH, audio_path=AUDIO_PATH)

# Transcribe
transcription = transcriber.transcribe(audio_path=AUDIO_PATH)

# CRITICAL: Free up VRAM before loading CLIP
print("Freeing Whisper VRAM...")
del transcriber
gc.collect()
torch.cuda.empty_cache() 

# --- PHASE 2: VIDEO (CLIP) ---
print("--- STARTING PHASE 2: VIDEO ---")
# Load CLIP only NOW, when Whisper is gone
ingestor = VideoIngestor(db_client=db) 

# Ingest Video
ingestor.process_video(
    video_path=VIDEO_PATH, 
    video_filename=VIDEO_NAME, 
    fps_sample_rate=1.0 
)

# Ingest Text (Lightweight, no GPU needed usually)
ingestor.process_transcript(
    segments=transcription['segments'], 
    video_filename=VIDEO_NAME
)

del ingestor
gc.collect()
torch.cuda.empty_cache()

print("--- PIPELINE COMPLETE ---")