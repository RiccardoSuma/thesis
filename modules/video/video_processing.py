import cv2
import os
import torch
import shutil
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from modules.vectorization.nomic_wrapper import NomicEmbedder

class VideoIngestor:
    def __init__(self, db_client):
        self.db = db_client
        self.model = None
        self.processor = None

    def _load_model_if_needed(self):
        if self.model is not None: return
        
        print("⚖️ Loading Qwen2-VL-2B (Native)...")
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.float16
            )
            self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=640*28*28)
            print("✅ Model Loaded.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def _generate_descriptions_native(self, frame_data_list):
        if not frame_data_list: return []
        self._load_model_if_needed()
        results = []
        
        # Prompt standard che mi ha dato buoni risultati
        prompt_text = "Analyze this slide. Extract title, charts/diagrams details, and key text."

        print(f"🚀 Processing {len(frame_data_list)} frames...")
        for item in tqdm(frame_data_list, desc="Inference"):
            image_path = item['image_path']
            try:
                image = Image.open(image_path)
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
                text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                
                item['vlm_description'] = output_text
                results.append(item)
                
                del inputs, generated_ids, image_inputs
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ Error on {image_path}: {e}")
                continue
        return results

    def process_video(self, video_path, video_filename, fps_sample_rate=0.5):
        # 1. CLEANUP INIZIALE FORZATO
        # Rimuove residui di crash precedenti
        temp_dir = "temp_frames"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = int(video_fps / fps_sample_rate)
        
        frames_to_process = []
        last_frame_gray = None

        print(f"--- 🎬 Processing {video_filename} ---")
        pbar = tqdm(total=total_frames, desc="Check & Extract")

        skipped_count = 0

        for current_frame in range(0, total_frames, frame_skip):
            timestamp = round(current_frame / video_fps, 2)

            # --- CHECK ESISTENZA (Salva tempo) ---
            if self.db.point_exists({"source": video_filename, "timestamp": timestamp, "modality": "visual"}):
                pbar.update(frame_skip)
                skipped_count += 1
                continue

            # Se non esiste, leggo il frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret: break
            
            # Change Detection Semplice
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            if last_frame_gray is not None:
                score = cv2.absdiff(gray, last_frame_gray).mean()
                if score < 6.0: 
                    pbar.update(frame_skip)
                    continue
            last_frame_gray = gray

            # Resize e Salvataggio
            h, w = frame.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            frame_path = os.path.join(temp_dir, f"{video_filename}_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)

            frames_to_process.append({
                "source": video_filename,
                "timestamp": timestamp,
                "image_path": frame_path
            })
            pbar.update(frame_skip)
        
        cap.release()
        pbar.close()

        if skipped_count > 0:
            print(f"⏩ Saltati {skipped_count} frame già presenti in Qdrant.")

        if not frames_to_process:
            print("✅ Nessun NUOVO frame da processare.")
            # Cleanup anche se vuoto
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            return

        # --- INFERENZA ---
        described_frames = self._generate_descriptions_native(frames_to_process)

        # --- UNLOAD MODEL ---
        if self.model:
            print("🗑️ Unloading Qwen2-VL...")
            del self.model
            del self.processor
            self.model = None
            torch.cuda.empty_cache()

        # --- EMBEDDING ---
        print("🧬 Embedding...")
        embedder = NomicEmbedder(device="cuda") 
        vectors, payloads = [], []

        for item in tqdm(described_frames, desc="Uploading"):
            text_content = item.get('vlm_description', "")
            if not text_content: continue
            
            vector = embedder.embed_document(text_content)
            payload = {
                "type": "smart_slide", "modality": "visual",
                "source": item['source'], "timestamp": item['timestamp'],
                "text": text_content, "image_path": item['image_path'],
                "is_chunk": False
            }
            vectors.append(vector)
            payloads.append(payload)

        if vectors:
            self.db.upload_batch(vectors, payloads)
            print(f"✅ Caricati {len(vectors)} nuovi vettori.")
        
        embedder.free_vram()
        
        # --- CLEANUP FINALE ---
        print("🧹 Cleaning temp frames...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # -- Transcript function invariata --
    def process_transcript(self, segments, video_filename):
        print(f"--- 🎙️ Ingesting Transcript: {video_filename} ---")
        embedder = NomicEmbedder(device="cuda")
        batch_vectors, batch_payloads = [], []
        for seg in tqdm(segments, desc="Audio Embedding"):
            full_text = seg['text'].strip()
            if not full_text: continue
            vector = embedder.embed_document(full_text)
            payload = { "type": "transcript_segment", "modality": "audio", "source": video_filename, "timestamp": round(seg['start'], 2), "text": full_text, "is_chunk": False }
            batch_vectors.append(vector)
            batch_payloads.append(payload)
            if len(batch_vectors) >= 64:
                self.db.upload_batch(batch_vectors, batch_payloads)
                batch_vectors, batch_payloads = [], []
        if batch_vectors: self.db.upload_batch(batch_vectors, batch_payloads)
        embedder.free_vram()
