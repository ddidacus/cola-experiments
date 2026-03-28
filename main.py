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
from .cola import cola_v2

def get_hparams(game):
    GAMMA = 0.96 #@param {type:"number"}
    NUM_RUNS =  10#@param {type:"number"}
    NUM_EPOCHS =  500#@param {type:"number"}
    ALPHA =  5.0#@param {type:"number"}
    BETA =   5.0#@param {type:"number"}
    SMOOTHING = 0.99

    if game == 'IPD':
        INPUT_DIM = 10
        STD = 0.1
    else:
        INPUT_DIM = 2
        STD =  1.0

    if game in ['Tandem', 'Balduzzi', 'Hamiltonian']:
        BATCH_SIZE =  8
        NUM_INNERLOOP_SHORT =  120000
        NUM_INNERLOOP_LONG =  120000
        NUM_NODES =  8
        interval = 1
        LR_SCHEDULER = 0.8
        LR = 1e-1
    else:
        BATCH_SIZE =  64
        NUM_INNERLOOP_SHORT =  80000
        NUM_INNERLOOP_LONG =  80000
        NUM_NODES =  16
        interval = 7
        LR_SCHEDULER = 1.0
        LR=0.001

    OUTPUT_DIM=INPUT_DIM//2

    return {
        "gamma": GAMMA,
        "num_runs": NUM_RUNS,
        "num_epochs": NUM_EPOCHS,
        "alpha": ALPHA,
        "std": STD,
        "batch_size": BATCH_SIZE,
        "num_innerloop_short": NUM_INNERLOOP_SHORT,
        "num_innerloop_long": NUM_INNERLOOP_LONG,
        "num_nodes": NUM_NODES,
        "beta": BETA,
        "interval": interval,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "lr_scheduler": LR_SCHEDULER,
        "lr": LR,
        "smoothing": SMOOTHING
    }

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

