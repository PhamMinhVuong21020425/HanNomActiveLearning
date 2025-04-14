# In this case, just configure for testing on CPU
# In reality, these parameters need to be configured appropriately

# Paths
HOME = '.'
DATA_DIR = f'{HOME}/data/wb_recognition_dataset'
WEIGHT_DIR = f'{HOME}/weights'
# KAGGLE_DIR = f'{HOME}/weights' # For local testing
KAGGLE_DIR = '/kaggle/input/effnetdropout/pytorch/default/1' # For Kaggle testing

# Hyperparams
SEED = 2025
DEVICE = "cpu"
N_ROUNDS = 2
N_EPOCHS = 3
N_INFER = 2
N_SAMPLES = 50
BATCH_SIZE = 16
TRAIN_BATCH = 32
VAL_BATCH = 64
PREDICT_BATCH = 64
N_INIT_LABELED = 50
N_SAMPLES_PREDICT = 200

# For optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
N_WORKERS = 0

# For WanDB
WANDB_PROJECT = "EfficientNetB7 Dropout Active Learning"
WANDB_KEY = "831a1da06e76c4aec094f6eca8c520482472a780"
WANDB_HOST = "https://api.wandb.ai/"
