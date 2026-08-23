# AUTO-GENERATED from spear-hall-of-fame.json - do not edit.
# Champion KV-cache eviction policy discovered by SPEAR.
import math

import torch

def spear_kv_policy(S, A, R):
    # Evolved by SPEAR â€” zero transcendental ops (exp/erf/tanh free)
    return ((4.5 * S) + (A + R))

def score(A, P, S, R):
    return spear_kv_policy(A, P, S, R)
