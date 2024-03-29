import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1E-8
eps = 1e-8
LR = 0.001
BATCH_SIZE = 64
NUMBER_OF_TRIALS = 5
SMALL_NORB_PATH = '../../data/'
