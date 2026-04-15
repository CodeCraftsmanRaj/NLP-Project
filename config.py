import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 3

MAX_LEN = 128
BATCH_SIZE = 16

LR = 2e-5
EPOCHS = 5

XLEX_PATH = "xlex.json"
XLEX_DIM = 6
MIN_WORD_FREQ = 5
XLEX_BUILD_SAMPLES = 3000