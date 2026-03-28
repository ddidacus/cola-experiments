#!/bin/bash

CONFIGS=(
    "configs/matching_pennies.yaml"
    "configs/ultimatum.yaml"
    "configs/ipd.yaml"
)

ALPHAS=(0.1 0.25 0.5 0.75 1.0 5.0)

for config in "${CONFIGS[@]}"; do
    game=$(basename "$config" .yaml)
    for alpha in "${ALPHAS[@]}"; do
        run_id="alpha_${alpha}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"

        sbatch <<EOF
#!/bin/bash
#SBATCH -J RR_alpha${alpha}_${game}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=long
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

source .venv/bin/activate

python exp_roundrobin.py --config ${config} --alpha ${alpha} --results_dir results/ --run_id ${run_id}
EOF
        echo "Submitted: config=${config} alpha=${alpha} run_id=${run_id}"
    done
done
