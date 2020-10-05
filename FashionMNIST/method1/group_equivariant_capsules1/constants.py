import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1E-8
LR = 0.0001
BATCH_SIZE = 64
NUMBER_OF_TRIALS = 1
SMALL_NORB_PATH = '../../data/'
