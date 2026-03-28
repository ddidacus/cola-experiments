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
        run_dir="results/${game}/${run_id}"

        sbatch <<EOF
#!/bin/bash
#SBATCH -J COLA_alpha${alpha}_${game}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=long
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out

source .venv/bin/activate

python exp_training.py --config ${config} --alpha ${alpha} --results_dir results/ --run_id ${run_id}

python exp_gradient_fields.py --config ${config} --results_dir ${run_dir} --output_dir ${run_dir}/gradient_fields

python exp_similarity.py --config ${config} --results_dir ${run_dir} --output_dir ${run_dir}/similarity
EOF
        echo "Submitted: config=${config} alpha=${alpha} run_id=${run_id}"
    done
done
