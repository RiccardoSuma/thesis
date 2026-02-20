import torch
import gc
import os
from modules.storage.qdrant_ops import VectorDB
from modules.video.video_processing import VideoIngestor
from modules.audio.video_transcribe import videoTranscriber

class Embedder:
    def __init__(self, collection_name="video_collection"):
        self.db = VectorDB(collection_name=collection_name)
        self.transcriber = None
        self.ingestor = None

    def _clear_vram(self, model_attr):
        model = getattr(self, model_attr)
        if model is not None:
            print(f"--- Clearing {model_attr} from VRAM ---")
            # For some models, calling .to('cpu') before del helps release memory faster
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model.to('cpu')
            
            del model
            setattr(self, model_attr, None)
            gc.collect()
            torch.cuda.empty_cache()

    def process_single_video(self, video_path, audio_tmp_path, fps=1.0):
        video_name = os.path.basename(video_path)
        try:
            # PHASE 1: AUDIO
            print(f"--- STARTING PHASE 1: AUDIO [{video_name}] ---")
            self.transcriber = videoTranscriber()
            self.transcriber.extract_audio_from_video(video_path, audio_tmp_path)
            # Transcription logic + JSON caching happens here
            transcription = self.transcriber.transcribe(audio_tmp_path, video_name)
            self._clear_vram('transcriber')

            # PHASE 2: VIDEO
            print(f"--- STARTING PHASE 2: VIDEO [{video_name}] ---")
            self.ingestor = VideoIngestor(db_client=self.db)
            # Cycle 1: Visual Ingestion (Frames + OCR)
            print(f"📸 Indexing Visuals for {video_name}...")
            self.ingestor.process_video(video_path, video_name, fps)
            
            # Cycle 2: Spoken Ingestion (Audio Segments)
            print(f"🎙️ Indexing Audio Transcript for {video_name}...")
            self.ingestor.process_transcript(transcription['segments'], video_name)
                
            self._clear_vram('ingestor')
            

            if os.path.exists(audio_tmp_path):
                os.remove(audio_tmp_path)
                print(f"🧹 Temporary audio removed: {audio_tmp_path}")

        except Exception as e:
            print(f"FAILED to process {video_name}: {str(e)}")
            self._clear_vram('transcriber')
            self._clear_vram('ingestor')