import whisper
import torch
import json
import os
import ffmpeg # Standard wrapper for ffmpeg-python

"""
video_transcribe module for extracting audio from a video.
Outputs a dict of text and segments with timestamps.
Optimized for Linux Worker Node performance.
"""

class videoTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Turbo is excellent for your 13-lecture scale
        self.model = whisper.load_model("turbo", device=self.device)
        self.cache_dir = "transcripts_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def transcribe(self, audio_path, video_filename):
        """
        Transcribes audio with persistence logic. 
        If a JSON cache exists, it returns immediately.
        """
        cache_path = os.path.join(self.cache_dir, f"{video_filename}.json")
        
        # 1. Check Cache (Crucial for restarts)
        if os.path.exists(cache_path):
            print(f"📦 Found cached transcript for {video_filename}. Skipping Whisper...")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f'Using device: {self.device}')
        print(f"Starting transcription of {video_filename}...")
        
        # 2. Linux Node Optimization: Use FP16 for Speed
        # Only use False if you get NaN errors, but on A100/RTX nodes, True is 2x-3x faster.
        use_fp16 = True if self.device == "cuda" else False
        
        try:
            result = self.model.transcribe(
                audio_path,
                language="it",     # Forced to Italian for speed
                fp16=use_fp16,
                verbose=False,     # Keep stdout clean for main's progress bar
                temperature=0,     # Greedy decoding for lecture stability
                beam_size=5
            )
            
            # 3. Save to JSON Cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                
            print(f"✅ Transcription Complete and cached.")
            return result

        except Exception as e:
            print(f"❌ Whisper Error: {e}")
            raise e

    def extract_audio_from_video(self, video_path, audio_path):
        """
        Extracts 16kHz mono WAV for Whisper.
        """
        # If the wav is already there (from a failed run), skip extraction
        if os.path.exists(audio_path):
            return

        print("Extracting audio from video...")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(
                    audio_path,
                    format='wav',
                    acodec='pcm_s16le',
                    ac=1,
                    ar='16000'
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            print("✅ Audio extraction complete.")
        except ffmpeg.Error as e:
            print(f"❌ FFmpeg Error: {e.stderr.decode()}")
            raise e