import torch
import pandas as pd
import numpy as np

from modles


if __name__ == '__main__':
    ckpt = torch.load('/home/tyk/bio_science/drug_generation/vae_denovo/ckpt', weight_only=False)
    LangVAE.load_state_dict(ckpt['model'])