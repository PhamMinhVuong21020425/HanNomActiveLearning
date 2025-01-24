import os
import random
import wandb

import numpy as np
import torch

# Apply WanDB
import wandb
from hyperparams import *

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Reconfig your WANDB API Key here
os.environ["WANDB_API_KEY"] = WANDB_KEY
os.environ["WANDB_BASE_URL"] = WANDB_HOST

# Login wandb
wandb.login()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
