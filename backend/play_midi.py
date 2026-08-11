import pygame
import time

def play_midi(file_path):
    pygame.mixer.init()
    
    pygame.mixer.music.load(file_path)
    
    print(f"Playing: {file_path}")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(1)
play_midi("./backend/data/output/athanasia.mid")