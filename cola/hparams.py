
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