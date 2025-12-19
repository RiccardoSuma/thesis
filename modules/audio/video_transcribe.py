import whisper
import torch
from ffmpeg import _ffmpeg, _run



"""
video_transcribe module for extracting audio from a video, the
final output is a dict of text and text segments with timestamps, useful
for selective CLIP embedding.
"""

class videoTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model=whisper.load_model("turbo", device=self.device)

    def transcribe(self, audio_path):
        print('Using device:', self.device)
        print("Starting transcription of audio track...")
        result = self.model.transcribe(
            audio_path,
            language="it", # Can eventually be detected automatically
            fp16=False, # FIX: Prevents the NaN error on some GPUs
            verbose=True # OPTIONAL: Prints progress in real-time
        )
        print("\nTranscription Complete.")
        return result

    

    def extract_audio_from_video(self, video_path, audio_path):
        print("Extracting audio from video...")
        stream = (
            _ffmpeg
            .input(video_path)
            .output(
                audio_path,
                format='wav',
                acodec='pcm_s16le',
                ac=1,
                ar=16000
            ).overwrite_output()
        )
        _run.run(stream)
        print("\nAudio extraction complete.")





