import os
import esm
import torch

# Set torch hub cache directory to avoid repeated downloads
# Must be set via TORCH_HUB_DIR environment variable
HUB_DIR = os.environ.get('TORCH_HUB_DIR')
if HUB_DIR:
    os.makedirs(HUB_DIR, exist_ok=True)
    torch.hub.set_dir(HUB_DIR)


def get_base_model(model_name):
    if model_name == 'ESM2_650M':
        model = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == 'ESM2_150M':
        model = esm.pretrained.esm2_t30_150M_UR50D()
    elif model_name == 'ESM2_35M':
        model = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == 'ESM2_8M':
        model = esm.pretrained.esm2_t6_8M_UR50S()
    elif model_name == 'ESM2_3B':
        model = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == 'ESM2_15B':
        model = esm.pretrained.esm2_t48_15B_UR50D()
    else:
        raise ValueError('Model name not recognized')
    
    return model
