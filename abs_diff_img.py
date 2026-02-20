import cv2
import numpy as np
import sys

def calculate_frame_diff(img1_path, img2_path):
    # 1. Caricamento
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("❌ Errore: Uno dei file non è stato trovato.")
        return

    # 2. Pre-processing
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize a 64x64: fondamentale per ignorare il rumore del codec video
    small1 = cv2.resize(gray1, (64, 64))
    small2 = cv2.resize(gray2, (64, 64))

    # 3. Calcolo Differenza
    diff = cv2.absdiff(small1, small2)
    mae = diff.mean()

    print(f"\n--- Analisi Differenza ---")
    print(f"Immagine 1: {img1_path}")
    print(f"Immagine 2: {img2_path}")
    print(f"Valore MAE (Soglia): {mae:.4f}")
    
    if mae < 3.0:
        print("Stato: Sotto soglia (Verranno considerate IDENTICHE)")
    elif mae < 7.0:
        print("Stato: Transizione leggera (Possibile rumore o micro-cambiamento)")
    else:
        print("Stato: Sopra soglia (Cambio slide RILEVATO)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python test_diff.py img1.jpg img2.jpg")
    else:
        calculate_frame_diff(sys.argv[1], sys.argv[2])