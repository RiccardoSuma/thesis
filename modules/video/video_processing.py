import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm

class VideoIngestor:
    def __init__(self, db_client, model_name='ViT-B/32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.db = db_client
        
    def process_video(self, video_path, video_filename, fps_sample_rate=1):
        """
        Reads video, extracts 1 frame every 'fps_sample_rate' seconds, 
        encodes them, and stores in Qdrant.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) # Source video FPS (e.g., 30 or 60)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate how many frames to skip to get 1 extraction per second
        frame_skip = int(fps / fps_sample_rate)
        
        current_frame = 0
        batch_vectors = []
        batch_payloads = []
        BATCH_SIZE = 32 # Send to Qdrant every 32 frames

        """
        Should BATCH_SIZE become too large (>>50), consider updating the upload_batch 
        function in qdrant_ops.py from using 'upsert' to 'upload' method for better performance
        """

        print(f"Starting ingestion: {video_filename} ({total_frames} frames)")
        
        # Progress bar
        pbar = tqdm(total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process if we hit the sample rate (e.g., every 30th frame)
            if current_frame % frame_skip == 0:
                # 1. Preprocess Image (OpenCV BGR -> PIL RGB)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                # 2. CLIP Encode
                image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 3. Convert to List
                vector = image_features.cpu().numpy()[0].tolist()
                
                # 4. Prepare Metadata
                timestamp = current_frame / fps
                payload = {
                    "type": "video_frame",
                    "source": video_filename,
                    "timestamp": timestamp,
                    "text": "" # Empty for frames
                }
                
                # 5. Add to Batch
                batch_vectors.append(vector)
                batch_payloads.append(payload)
                
                # 6. Upload if batch is full
                if len(batch_vectors) >= BATCH_SIZE:
                    self.db.upload_batch(batch_vectors, batch_payloads)
                    batch_vectors = []
                    batch_payloads = []

            current_frame += 1
            pbar.update(1)

        # Upload remaining frames
        if batch_vectors:
            self.db.upload_batch(batch_vectors, batch_payloads)
            
        cap.release()
        print("Video ingestion complete.")

    def process_transcript(self, segments, video_filename):
        """
        Ingests Whisper segments (Text) into the SAME collection.
        segments: List of dicts [{'start': 0.5, 'text': 'Hello world'}]
        """
        batch_vectors = []
        batch_payloads = []
        BATCH_SIZE = 32

        print("Ingesting Transcripts...")

        pbar = tqdm(total=len(segments))

        
        for seg in segments:
            text = seg['text']
            timestamp = seg['start'] # Start time of the sentence
            
            # 1. CLIP Encode Text
            text_tokens = clip.tokenize([text[:77]]).to(self.device) # CLIP limit 77 tokens
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 2. Convert
            vector = text_features.cpu().numpy()[0].tolist()
            
            # 3. Payload
            payload = {
                "type": "transcript_segment",
                "source": video_filename,
                "timestamp": timestamp,
                "text": text
            }
            
            batch_vectors.append(vector)
            batch_payloads.append(payload)

            if len(batch_vectors) >= BATCH_SIZE:
                self.db.upload_batch(batch_vectors, batch_payloads)
                batch_vectors = []
                batch_payloads = []
            
            pbar.update(1)

        if batch_vectors:
            self.db.upload_batch(batch_vectors, batch_payloads)
        print("Transcript ingestion complete.")