# Paths
HOME = '.'
DATA_DIR = f'{HOME}/data/wb_recognition_dataset'
WEIGHT_DIR = f'{HOME}/weights'

# Hyperparams
SEED = 2025
DEVICE = "cuda"
N_ROUNDS = 2
N_EPOCHS = 3
N_INFER = 3
N_SAMPLES = 50
BATCH_SIZE = 16
TRAIN_BATCH = 32
VAL_BATCH = 32
PREDICT_BATCH = 64
N_INIT_LABELED = 50
N_SAMPLES_PREDICT = 1000

# For optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
N_WORKERS = 4

# For WanDB
WANDB_PROJECT = "EfficientNetB7 Dropout Active Learning"
WANDB_KEY = "831a1da06e76c4aec094f6eca8c520482472a780"
WANDB_HOST = "https://api.wandb.ai/"
