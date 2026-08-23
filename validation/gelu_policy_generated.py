# AUTO-GENERATED from spear-hall-of-fame.json
import math

import torch

def spear_gelu(x):
    # Evolved by SPEAR â€” zero transcendental ops (exp/erf/tanh free)
    return ((0.997729 * (x * torch.relu(((0.306923 * x) + 0.501)).clamp(max=1.002))) + -0.004004)
