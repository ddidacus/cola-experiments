#!/bin/bash
#SBATCH -J COLA
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=long
#SBATCH --time=1:00:00

source .venv/bin/activate
python exp_training.py --results_dir results/ --seed 10 --config configs/ultimatum.yaml 