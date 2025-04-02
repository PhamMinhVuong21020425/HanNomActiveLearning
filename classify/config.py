import os
import random
import wandb
import numpy as np
import torch

from .hyperparams import *

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# No warnings
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Reconfig your WANDB API Key here
os.environ["WANDB_API_KEY"] = WANDB_KEY
os.environ["WANDB_BASE_URL"] = WANDB_HOST

# Login wandb
if wandb.api.api_key is None:
    wandb.login()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
