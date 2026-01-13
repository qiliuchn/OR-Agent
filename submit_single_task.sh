#!/bin/bash
#SBATCH --job-name=oragent_single_task_tsp_constructive2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=120:00:00


# =====Tasks=====
# algorithm options: oragent, reevo, eho, ael, funsearch
# problem options: tsp_constructive, tsp_aco, tsp_gls, tsp_pomo, tsp_lehd
# cvrp_aco, cvrp_pomo, cvrp_lehd, bpp_online, bpp_offline_aco
# op_aco, mkp_aco, dpp_ga, driving
# TODO: change algorithm and problem settings
ALGORITHM="oragent"
PROBLEM="tsp_constructive"


# =====Load env on Tongji HPC=====
# Activate the Conda environment
# Note: conda init bash should be done once during setup, not in job scripts
# Instead, source the conda initialization that's already in .bashrc
source ~/.bashrc
conda activate sumo

# Load relevant modules
# Load SUMO module for "driving problem"
module load xerces-c/3.2.3 sumo/1.20

# Change directory
cd /share/home/u23310103/apps/or_agent


# Output directory
# TODO: carefully set output dir so that tasks don't overwrite each other!
OUTPUT_DIR="outputs/${ALGORITHM}/${PROBLEM}/V2"
mkdir -p "$OUTPUT_DIR"

# Print out info
echo "Running single task: $ALGORITHM on $PROBLEM"
echo "Output directory: $OUTPUT_DIR"


# Run the task  TODO: check cmd 
python -u src/oragent/cli.py \
    --algorithm "$ALGORITHM" \
    --problem "$PROBLEM" \
    --max-evolutions 500 \
    --elitist-as-root-period 2 \
    --num-children 2 \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/output.txt" 2>&1

# How to use
#```bash
# sbatch submit_combinatorial_tasks.sh
#```