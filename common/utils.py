"""----------------------------------------------------
Define various functional functions.
----------------------------------------------------"""

import numpy as np
import random
import os

import torch

def preprocess_state(state):
    """
    Processes the observation dictionary from the wrapper.
    - Extracts the 'image' array.
    - Normalizes pixel values to [0, 1].
    - Permutes dimensions from (H, W, C) to (C, H, W).
    """
    state = state['image'].transpose((2, 0, 1))
    return state.astype(np.float32) / 255.0

def all_seed(seed = 1):
    ''' omnipotent seed for RL, attention the position of seed function, you'd better put it just following the env create function
    Args:
        env (_type_): 
        seed (int, optional): _description_. Defaults to 1.
    '''
    if seed == 0:
        return
    # print(f"seed = {seed}")
    # env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # config for GPU
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
        # config for cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
