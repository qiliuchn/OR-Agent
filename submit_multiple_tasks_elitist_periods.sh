#!/bin/bash
#SBATCH --job-name=oragent_multiple_tasks_elitist_periods
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3                         # Manual: TODO: Update when changing algorithms/problems

# Ablation studies
# Configuration for each task  TODO: change
ALGORITHM="oragent"
PROBLEM="op_aco"
NUM_CHILDREN=2

# Tasks in this file vary "--elitist-as-root-period" settings
# CONFIG format: <elitist-as-root-period>
CONFIG=(
    0
    1
    2
    3
)

# Calculate total tasks
TOTAL_TASKS=${#CONFIG[@]}

# Get task ID with fallback
TASK_ID=${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ARRAY_INDEX:-0}}


# Parse configuration for this array task TODO: use right way to parse
ELITIST_AS_ROOT_PERIOD=${CONFIG[$TASK_ID]}


# Output directory
# TODO: carefully set output dir so that tasks don't overwrite each other!
OUTPUT_DIR="outputs/${ALGORITHM}/${PROBLEM}/ELITIST_AS_ROOT_PERIOD_${ELITIST_AS_ROOT_PERIOD}"
mkdir -p "$OUTPUT_DIR"

echo "TASK_ID: $TASK_ID, SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID, SLURM_JOB_ARRAY_INDEX: $SLURM_JOB_ARRAY_INDEX"
echo "Total tasks: $TOTAL_TASKS"
echo "Running task $TASK_ID: $ALGORITHM on $PROBLEM"
echo "Output directory: $OUTPUT_DIR"

# Load environment
source ~/.bashrc
conda activate sumo
module load xerces-c/3.2.3 sumo/1.20

# Change directory
cd /share/home/u23310103/apps/or_agent


# Run the task TODO: check cmd 
python -u src/oragent/cli.py \
    --algorithm "$ALGORITHM" \
    --problem "$PROBLEM" \
    --max-evolutions 500 \
    --num-children "$NUM_CHILDREN" \
    --elitist-as-root-period "$ELITIST_AS_ROOT_PERIOD" \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/output.txt" 2>&1