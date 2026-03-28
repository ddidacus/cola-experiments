# COLA: Consistent Learning with Opponent-Learning Awareness
Accompanying repository for COLA: Consistent Learning with Opponent-Learning Awareness

Run COLA and other general-sum learning algorithms on a set of zero- and general-sum games.

Please raise issues if you encounter any problems!

## Setup

Each game has a YAML config file under `configs/`:

```
configs/
  matching_pennies.yaml
  ipd.yaml
  ultimatum.yaml
  chicken_game.yaml
  tandem.yaml
  balduzzi.yaml
  hamiltonian.yaml
```

A config file specifies game hyperparameters and the available algorithms. Algorithms are a list; each entry has a `name`, `beta` (lookahead rate), and `enabled` flag. Duplicate entries (e.g. two COLA_long with different betas) are supported.

```yaml
game: "Matching Pennies"

gamma: 0.96
num_epochs: 500
alpha: 5.0          # outer learning rate
beta: 0.5           # default lookahead rate
la_rates: [0.5, 10.0]  # LA-rates for COLA pre-training (plots all, uses smallest for training)
smoothing: 0.99

input_dim: 2
std: 1.0
batch_size: 64
num_innerloop_long: 8000
num_nodes: 16
interval: 7
lr_scheduler: 1.0
lr: 0.001

algorithms:
  - name: COLA_long
    beta: 5.0
    enabled: true
  - name: COLA_short
    beta: 5.0
    enabled: true
  - name: LOLA
    beta: 5.0
    enabled: true
  - name: HOLA4
    beta: 5.0
    enabled: true
  - name: CGD
    beta: 5.0
    enabled: true
  - name: SOS
    beta: 5.0
    enabled: false
```

### Supported algorithms

| Name | Description |
|------|-------------|
| `COLA_long` | Neural net approximation of lookahead gradients, pre-trained for `num_innerloop_long` steps |
| `COLA_short` | Same as COLA_long but hardcoded to 800 pre-training steps |
| `LOLA` | Learning with Opponent-Learning Awareness |
| `HOLA<N>` | Higher-Order LOLA with order N (e.g. `HOLA1`, `HOLA4`, `HOLA8`) |
| `CGD` | Competitive Gradient Descent |
| `SOS` | Stable Opponent Shaping |

## Command-line scripts

Results are stored under `--results_dir` (default: `results/`), organized as `results/<game>/<run_id>/`. Each run folder contains a copy of the config used, plots, and network weights.

### 1. Train COLA and baselines

```bash
# Run all enabled algorithms from the config
python exp_training.py --config configs/matching_pennies.yaml

# Override alpha (outer learning rate)
python exp_training.py --config configs/matching_pennies.yaml --alpha 0.5

# Specify number of independent runs (default: 10)
python exp_training.py --config configs/matching_pennies.yaml --num_runs 5

# Set a custom run folder name (default: timestamp)
python exp_training.py --config configs/matching_pennies.yaml --run_id my_experiment

# Load pretrained COLA weights from a previous run
python exp_training.py --config configs/matching_pennies.yaml --load_path results/matching_pennies/20260328_120000
```

Arguments:
- `--config` (required): path to YAML config file
- `--alpha`: override alpha (outer learning rate) from config
- `--num_runs`: number of independent runs per algorithm, each with a different random seed (default: 10)
- `--load_path`: path to a previous run folder to load pretrained COLA weights from
- `--results_dir`: base output directory (default: `results/`)
- `--run_id`: run folder name (default: timestamp)

Saves to `results/<game>/<run_id>/`:
- `config.yaml` -- copy of the config used
- `seeds.yaml` -- random seeds used for each run
- `h_net_la<rate>.pth`, `k_net_la<rate>.pth` -- trained COLA_long networks per LA-rate
- `h_net_short.pth`, `k_net_short.pth` -- trained COLA_short networks (if enabled)
- `total_losses.npy` -- pretraining consistency loss (best LA-rate)
- `avg_loss.png` -- average loss curves per algorithm
- `log_consistency_loss.png` -- COLA pretraining loss per LA-rate (log scale)
- `variance_over_consistency.png` -- scatter plot of variance vs consistency loss (log-log)

### 2. Evaluate consistency loss

Requires trained COLA networks from a previous `exp_training.py` run.

```bash
python exp_consistency_loss.py --config configs/matching_pennies.yaml --results_dir results/matching_pennies/<run_id>
```

Saves `consistency_loss.json` with metrics (dot, norm, L2, squared_diff) for COLA, HOLA1/2/3, SOS, and CGD.

### 3. Plot gradient fields

Requires trained COLA networks from a previous `exp_training.py` run.

```bash
python exp_gradient_fields.py --config configs/matching_pennies.yaml --results_dir results/matching_pennies/<run_id> --output_dir results/matching_pennies/<run_id>/gradient_fields
```

Saves gradient field plots:
- `cola_gradient_field.png`, `cola_errors_heatmap.png`, `cola_x_components_heatmap.png`, `cola_y_components_heatmap.png`
- `lola_streamplot.png`, `lola_gradient_field.png`
- `hola4_gradient_field.png`

### 4. Measure similarity between COLA and HOLA/SOS

Requires trained COLA networks from a previous `exp_training.py` run.

```bash
python exp_similarity.py --config configs/matching_pennies.yaml --results_dir results/matching_pennies/<run_id> --output_dir results/matching_pennies/<run_id>/similarity
```

Saves `similarity.json` with cosine similarity between COLA's update direction and HOLA1/4/6 and SOS.

### 5. Round-robin: COLA vs each opponent

Trains each COLA variant against every non-COLA algorithm in asymmetric pairs: player 0 uses COLA, player 1 uses the opponent method.

```bash
# Run round-robin for all enabled algorithms in the config
python exp_roundrobin.py --config configs/matching_pennies.yaml

# Load pre-trained COLA weights from a previous exp_training.py run
python exp_roundrobin.py --config configs/matching_pennies.yaml --load_path results/matching_pennies/<run_id>
```

Saves to `results/<game>/roundrobin_<run_id>/`:
- `<COLA>_vs_<opponent>/avg_loss.png` -- per-pair loss curves for both players
- `summary_cola_loss.png` -- COLA's loss across all opponents
- `summary_opponent_loss.png` -- each opponent's loss when paired with COLA
- COLA network weights (same as `exp_training.py`)

### 6. Alpha sweep (SLURM)

Submit a batch of jobs sweeping over alpha values across multiple games:

```bash
bash run_alpha_sweep.sh
```

This submits SLURM jobs that run `exp_training.py`, `exp_gradient_fields.py`, and `exp_similarity.py` for each (game, alpha) combination.
