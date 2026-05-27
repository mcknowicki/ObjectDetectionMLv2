# konfiguracja generowanych obrazów i cech hog
IMG_SIZE = (128, 128)
PIXELS_PER_CELL = (4, 4) # najlepiej 4 4. 8 8 daje mniej cech, co przyspiesza operacje, 16, 16 jest bardziej dokładne, ale obciąża sprzęt
CELLS_PER_BLOCK = (2, 2) # najlepiej 2 2
NUM_ROTATIONS = 5 # liczba rotacji na obraz

# wybór danych wyjściowych
DATASET = 'roadsigns'
#DATASET = 'buildings'
SUFFIX = '1'
INPUT_DIR = f'./data/input/{DATASET}'

# konfiguracja zakłóceń
NOISE_STD = 0.15 # poziom szumów
BLUR_SIGMA = 2.0 # poziom rozmycia
ENABLE_OCCLUSION = True # trigger losowych zasłonięć obrazów

# prawdopodobieństwa wystąpienia zakłóceń
NOISE_PROBABILITY = 0.8
BLUR_PROBABILITY = 0.6
OCCLUSION_PROBABILITY = 0.6

# trigger wyświetlający błędnie zaklasyfikowane obrazy w summary
SHOW_FALSE_PREDICTIONS = False