import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 128

class VideoIngestor:
    def __init__(self, db_client, model_name='ViT-B/32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.db = db_client 

        
        from modules.video.ocr import OCRProcessor
        self.ocr = OCRProcessor(languages=['en', 'it'])

    def process_video(self, video_path, video_filename, fps_sample_rate=1.0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = int(video_fps / fps_sample_rate)
        
        batch_vectors, batch_payloads = [], []
        last_frame_gray = None
        last_ocr_text = ""

        print(f"--- 🎬 Ingestion Smart: {video_filename} ---")
        pbar = tqdm(total=total_frames, desc="Frames")

        for current_frame in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret: break
            
            timestamp = round(current_frame / video_fps, 2)
            
            # 1. CHECK ESISTENZA SMART
            # Uso chunk_index=0 come sentinella per il video
            if self.db.point_exists({"source": video_filename, "timestamp": timestamp, "modality": "visual", "chunk_index": 0}):
                pbar.update(frame_skip)
                continue

            # 2. CHANGE DETECTION (Soglia a 6.0 basata sui test)
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            skip_ocr = (last_frame_gray is not None and cv2.absdiff(gray, last_frame_gray).mean() < 6.0)
            last_frame_gray = gray

            # 3. OCR CON RESIZING A 720p
            if not skip_ocr:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize a 720p per velocizzare l'estrazione mantenendo qualità
                img_resized = cv2.resize(img_rgb, (1280, 720), interpolation=cv2.INTER_AREA)
                ocr_text = self.ocr.extract_text(img_resized).strip()
                last_ocr_text = ocr_text
            else:
                ocr_text = last_ocr_text
            
            if not ocr_text:
                pbar.update(frame_skip)
                continue

            # 4. CHUNKING & ENCODING
            words = ocr_text.split()
            chunk_size = 60
            overlap = 15
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size - overlap)] if len(words) > chunk_size else [words]

            for idx, chunk_words in enumerate(chunks):
                text_chunk = " ".join(chunk_words)
                try:
                    tokens = clip.tokenize([text_chunk]).to(self.device)
                    with torch.no_grad():
                        features = self.model.encode_text(tokens)
                        features /= features.norm(dim=-1, keepdim=True)
                    
                    vector = features.cpu().numpy()[0].tolist()
                    
                    payload = {
                        "type": "video_frame",
                        "modality": "visual",
                        "source": video_filename,
                        "timestamp": timestamp,
                        "text": text_chunk,      
                        "full_ocr": ocr_text,    
                        "chunk_index": idx,         # Cruciale per l'ID unico
                        "total_chunks": len(chunks),
                        "is_chunk": True if len(chunks) > 1 else False
                    }
                    
                    batch_vectors.append(vector)
                    batch_payloads.append(payload)
                except: 
                    continue

            # Upload periodico
            if len(batch_vectors) >= BATCH_SIZE:
                self.db.upload_batch(batch_vectors, batch_payloads)
                batch_vectors, batch_payloads = [], []

            pbar.update(frame_skip)

        if batch_vectors:
            self.db.upload_batch(batch_vectors, batch_payloads)
        
        cap.release()
        pbar.close()
        
    def process_transcript(self, segments, video_filename):
        """
        Ingestione dei segmenti audio Whisper.
        Sincronizzato con la logica MD5 e modality: 'audio'.
        """
        print(f"--- 🎙️ Ingesting Transcript: {video_filename} ---")
        batch_vectors = []
        batch_payloads = []
        BATCH_SIZE = 32

        for seg in tqdm(segments, desc="Audio Segments"):
            full_text = seg['text'].strip()
            if not full_text: continue

            # CLIP Limit: Tkenizzo il testo (max 77 token gestiti internamente da clip.tokenize)
            # Uso i primi 60-70 termini per sicurezza
            text_to_encode = " ".join(full_text.split()[:65])
            
            try:
                tokens = clip.tokenize([text_to_encode]).to(self.device)
                with torch.no_grad():
                    # Encoding testuale (simmetrico alla query e alle slide OCR)
                    text_features = self.model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                
                vector = text_features.cpu().numpy()[0].tolist()
                
                payload = {
                    "type": "transcript_segment",
                    "modality": "audio",
                    "source": video_filename,
                    "timestamp": round(seg['start'], 2),
                    "text": full_text, # Salvo tutto il testo per il RAG
                    "is_chunk": False
                }
                
                batch_vectors.append(vector)
                batch_payloads.append(payload)

            except Exception as e:
                print(f"⚠️ Errore encoding segmento audio: {e}")
                continue

            if len(batch_vectors) >= BATCH_SIZE:
                self.db.upload_batch(batch_vectors, batch_payloads)
                batch_vectors, batch_payloads = [], []

        if batch_vectors:
            self.db.upload_batch(batch_vectors, batch_payloads)