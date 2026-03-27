import os, sys
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.fft as fft
import torch.nn as nn
from datetime import datetime
import seaborn as sns
from scipy import stats
from torch.nn import CosineSimilarity

from .networks import NonPGNet, PGNet
from .utils import (
    get_gradient, 
    init_th, 
    load_checkpoint, 
    smooth, 
    get_hessian, 
    update_th
)
from .hparams import get_hparams
from .cola import cola_v2

if __name__ == "__main__":

    cos = CosineSimilarity(dim=1, eps=1e-15)
    markers = ['o', 'v', '^', '*',  's', 'D']
    colors = ['blue', 'red', 'green', 'purple']
    sns.set_theme(style="ticks")
    fontsize=30
    game = 'Matching Pennies' #@param ["Matching Pennies", "IPD", "Ultimatum", "Chicken Game", "Tandem", "Balduzzi", "Hamiltonian"]

    hyper_params = get_hparams(game)
    network_type = PGNet if game in ['Tandem', 'Balduzzi', 'Hamiltonian'] else NonPGNet 
    network = network_type(hyper_params=hyper_params)

