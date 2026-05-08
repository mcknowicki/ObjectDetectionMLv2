# konfiguracja generowanych obrazów i ich cech
IMG_SIZE = (128, 128)
PIXELS_PER_CELL = (4, 4)
CELLS_PER_BLOCK = (2, 2) # PIXELS_PER_CELL (8, 8) daje mniej cech, co przyspiesza operacje na nich, (16, 16) jest bardziej dokładne
NUM_ROTATIONS = 5 # liczba rotacji na obraz

# wybór zbioru danych
DATASET = 'roadsigns'
#DATASET = 'buildings'
INPUT_DIR = f'./data/input/{DATASET}'

# konfiguracja
CORRUPTED_DATASET = True # znacznik generowania zakłóceń
CORRUPTION_PROBABILITY = 0.5
NOISE_STD = 0.08 # poziom zakłóceń
BLUR_SIGMA = 1.0
ENABLE_OCCLUSION = True # losowe zasłonięcie fragmentu
NOISE_PROBABILITY = 0.5
BLUR_PROBABILITY = 0.3
OCCLUSION_PROBABILITY = 0.2