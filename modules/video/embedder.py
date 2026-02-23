import torch
import gc
import os
import shutil
from modules.storage.qdrant_ops import VectorDB
from modules.video.video_processing import VideoIngestor
from modules.audio.video_transcribe import videoTranscriber

class Embedder:
    def __init__(self, collection_name="video_collection"):
        # Nota: VectorDB ora usa size=768 di default per Nomic
        self.db = VectorDB(collection_name=collection_name)
        self.transcriber = None
        self.ingestor = None

    def _clear_vram(self, component_name):
        """
        Forza bruta per liberare VRAM. 
        Essenziale quando si switcha da Whisper a Qwen a Nomic.
        """
        print(f"🧹 Clearing {component_name} from Memory...")
        
        if component_name == 'transcriber' and self.transcriber:
            if hasattr(self.transcriber, 'model'):
                del self.transcriber.model
            del self.transcriber
            self.transcriber = None
            
        if component_name == 'ingestor' and self.ingestor:
            # L'ingestor nuovo gestisce la sua memoria internamente (carica/scarica Qwen),
            # ma distruggerlo qui è una sicurezza in più.
            del self.ingestor
            self.ingestor = None

        gc.collect()
        torch.cuda.empty_cache()

    def process_single_video(self, video_path, audio_tmp_path, fps=0.5):
        video_name = os.path.basename(video_path)
        try:
            # --- FASE 1: AUDIO (Whisper) ---
            print(f"\n🎧 [Phase 1/2] Audio Processing: {video_name}")
            self.transcriber = videoTranscriber()
            self.transcriber.extract_audio_from_video(video_path, audio_tmp_path)
            
            # Trascrizione
            transcription = self.transcriber.transcribe(audio_tmp_path, video_name)
            
            # IMPORTANTE: Libero subito Whisper dalla VRAM
            self._clear_vram('transcriber')

            # --- FASE 2: VIDEO & HYBRID INGESTION (Qwen + Nomic) ---
            print(f"\n👁️ [Phase 2/2] Visual & Semantic Ingestion: {video_name}")
            self.ingestor = VideoIngestor(db_client=self.db)
            
            # A. Processo i Frame (Vision -> Text -> Vector)
            self.ingestor.process_video(video_path, video_name, fps_sample_rate=fps)
            
            # B. Processo l'Audio (Text -> Vector con Nomic)
            # Passo i segmenti trascritti al nuovo ingestor che usa Nomic
            if transcription and 'segments' in transcription:
                self.ingestor.process_transcript(transcription['segments'], video_name)
            
            self._clear_vram('ingestor')
            
            # Pulizia file temporanei
            if os.path.exists(audio_tmp_path):
                os.remove(audio_tmp_path)
            
            #Pulizia cartella temp_frames (se voglio risparmiare spazio disco)
            temp_frames_dir = os.path.join("modules", "video", "temp_frames")
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
                os.makedirs(temp_frames_dir, exist_ok=True)

            print(f"✅ SUCCESS: {video_name} fully ingested.")

        except Exception as e:
            print(f"❌ CRITICAL ERROR processing {video_name}: {str(e)}")
            self._clear_vram('transcriber')
            self._clear_vram('ingestor')
            raise e