IMG_SIZE = (128, 128)
PIXELS_PER_CELL = (4, 4)
CELLS_PER_BLOCK = (2, 2) # PIXELS_PER_CELL (8, 8) daje mniej cech, co przyspiesza operacje na nich, (16, 16) jest bardziej dokładne
NUM_ROTATIONS = 5 # liczba rotacji na obraz

DATASET = 'roadsigns'
#DATASET = 'buildings'
#DATASET = 'animals'

INPUT_DIR = f'./data/input/{DATASET}'