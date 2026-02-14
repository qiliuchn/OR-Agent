#!/bin/bash
#SBATCH --job-name=elitist_as_period_ablations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --array=0-4                         # Manual: TODO: Update when changing algorithms/problems
                                            # Current: 1 algorithms × 1 problem x 5 levels = 5 tasks (0-4)


# Algorithms and problems for all combinations TODO: change
ALGORITHM="oragent"
PROBLEM="cvrp_pomo"  # TODO: change the name of the problem you want to test

# Tasks in this file vary "--elitist-as-root-period" settings
# CONFIG format: <elitist-as-root-period>
CONFIG=(
    2
    4
    8
    16
    32
)

# Calculate algorithm and problem indices from array task ID
# SLURM_ARRAY_TASK_ID or SLURM_JOB_ARRAY_INDEX is an environment variable automatically set by SLURM for job arrays
# A unique number (0, 1, 2, ...) assigned to each task in a job array
# For #SBATCH --array=0-9, it will be 0 through 9
TASK_ID=${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ARRAY_INDEX:-0}}

# Parse configuration for this array task TODO: use right way to parse
ELITIST_AS_ROOT_PERIOD=${CONFIG[$TASK_ID]}


# Output directory
# TODO: carefully set output dir so that tasks don't overwrite each other!
OUTPUT_DIR="outputs/${ALGORITHM}/${PROBLEM}/elitist_as_period_${ELITIST_AS_ROOT_PERIOD}"
mkdir -p "$OUTPUT_DIR"


# Print out info
echo "TASK_ID: $TASK_ID"
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
    --init-pop-size 20 \
    --num-children 2 \
    --max-tree-depth 3 \
    --fast-exploration-for-crossover \
    --max-debug-rounds 2 \
    --max-experiment-repeats 3 \
    --elitist-as-root-period "$ELITIST_AS_ROOT_PERIOD" \
    --elitist-enlargement-factor 2.0 \
    --reflection-period 0 \
    --reflection-disabled-for-crossover \
    --reflection-elitist-synchro \
    --reflection-compression 100 \
    --timeout-seconds 300 \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/output.txt" 2>&1

# How to use
#```bash
# sbatch submit_combinatorial_tasks.sh
#```
# SLURM will create TOTAL_TASKS independent jobs (array tasks 0-(TOTAL_TASKS-1))
# Each gets 4 CPUs and 16GB RAM
# Each runs a different algorithm-problem combination
# Outputs go to separate files: oragent_combinatorial_tasks_<JOB_ID>_<TASK_ID>.out/.err
# Results go to results/<algorithm>_<problem>/output.txt