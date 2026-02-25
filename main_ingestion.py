import os
import subprocess
import platform
from modules.video.embedder import Embedder

def free_onedrive_space(file_path):
    """Gestisce lo spazio OneDrive su Windows (online-only)."""
    if platform.system() == "Windows":
        try:
            subprocess.run(['attrib', '+U', '-P', file_path], check=True)
            print(f"✅ OneDrive space freed for: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️ OneDrive purge failed: {e}")
    else:
        pass

def main():
    VIDEO_DIR = "/home/user0/Suma/ABB/"
    pipeline = Embedder(collection_name="abb_video_collection_v2")

    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

    if not video_files:
        print(f"❌ No .mp4 files found in {VIDEO_DIR}")
        return

    for video_file in video_files:
        full_path = os.path.join(VIDEO_DIR, video_file)
        

        check_payload = {
            "source": video_file, 
            "timestamp": 0.0, 
            "modality": "visual"
        }
        

        if pipeline.db.point_exists(check_payload):
            print(f"⏩ Skipping {video_file}: Already indexed.")
            continue


        audio_tmp = os.path.join(VIDEO_DIR, f"{video_file}_temp.wav")
        print(f"\n🚀 >>> Processing: {video_file}")
        
        try:
            pipeline.process_single_video(
                video_path=full_path,
                audio_tmp_path=audio_tmp,
                fps=1.0 
            )
        except Exception as e:
            print(f"❌ Failed to process {video_file}: {e}")
            continue

if __name__ == "__main__":
    main()