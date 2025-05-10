# config.py

# General Settings
SEED = 42
DEVICE = "cuda"  # or "cpu"

# Model Settings
IN_CHANNELS = 4  # MRI modalities for example
NUM_CLASSES = 3  # WT, TC, ET for Brain Tumor
EMBED_DIMS = [32, 64, 160, 256]
NUM_HEADS = [1, 2, 5, 8]
MLP_RATIOS = [4, 4, 4, 4]
SR_RATIOS = [4, 2, 1, 1]
DEPTHS = [2, 2, 2, 2]

# Decoder Settings
DECODER_CHANNELS = [256, 128, 64, 32]
DROPOUT_RATE = 0.5

# Training Settings
BATCH_SIZE = 1
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 300
SAVE_DIR = "./checkpoints"
PRED_SAVE_DIR = "./predictions"

# Scheduler Settings
SCHEDULER_MODE = "cosine"  # or "plateau"
T_MAX = 100  # for CosineAnnealingLR
MIN_LR = 1e-6
PATIENCE = 10  # for ReduceLROnPlateau

# Loss Settings
DICE_WEIGHT = 1.0
TVERSKY_WEIGHT = 1.0
CE_WEIGHT = 1.0
BOUNDARY_WEIGHT = 1.0
AUX_WEIGHT = 0.4
