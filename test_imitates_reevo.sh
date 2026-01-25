#!/bin/bash
#SBATCH --job-name=imitate_reevo2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --array=0-4                         # Manual: TODO: Update when changing algorithms/problems
                                            # Current: 1 algorithms × 5 problem = 5 tasks (0-4)

# =====================
# imitate_reevo2
# =====================
# num_children=1
# max_tree_depth=1
# max_debug_rounds=0
# max_experiment_repeats=0
# elitist_as_root_period=11
# elitist_enlargement_factor=5.0
# reflection_period=8
# reflection_compression=50
# reflection_disabled_for_crossover=True
# reflection_elitist_synchro=True

# Algorithms and problems for all combinations TODO: change
ALGORITHMS=("oragent")
PROBLEMS=("tsp_constructive" "cvrp_pomo" "dpp_ga" "mkp_aco" "op_aco")

# Calculate total tasks for array range
NUM_ALGORITHMS=${#ALGORITHMS[@]}
NUM_PROBLEMS=${#PROBLEMS[@]}
TOTAL_TASKS=$((NUM_ALGORITHMS * NUM_PROBLEMS))

# Calculate algorithm and problem indices from array task ID
# SLURM_ARRAY_TASK_ID or SLURM_JOB_ARRAY_INDEX is an environment variable automatically set by SLURM for job arrays
# A unique number (0, 1, 2, ...) assigned to each task in a job array
# For #SBATCH --array=0-9, it will be 0 through 9
TASK_ID=${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ARRAY_INDEX:-0}}
ALG_INDEX=$((TASK_ID / NUM_PROBLEMS))
PROB_INDEX=$((TASK_ID % NUM_PROBLEMS))

# Get algorithm and problem
ALGORITHM=${ALGORITHMS[$ALG_INDEX]}
PROBLEM=${PROBLEMS[$PROB_INDEX]}

# Output directory
# TODO: carefully set output dir so that tasks don't overwrite each other!
OUTPUT_DIR="outputs/${ALGORITHM}/${PROBLEM}/imitate_reevo2"
mkdir -p "$OUTPUT_DIR"

# Print out info
echo "TASK_ID: $TASK_ID, SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID, SLURM_JOB_ARRAY_INDEX: $SLURM_JOB_ARRAY_INDEX"
echo "Total algorithms: $NUM_ALGORITHMS, Total problems: $NUM_PROBLEMS, Total tasks: $TOTAL_TASKS"
echo "Running task $TASK_ID: $ALGORITHM on $PROBLEM"
echo "Algorithm index: $ALG_INDEX, Problem index: $PROB_INDEX"
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
    --num-children 1 \
    --max-tree-depth 1 \
    --max-debug-rounds 0 \
    --max-experiment-repeats 0 \
    --elitist-as-root-period 11 \
    --elitist-enlargement-factor 5.0 \
    --reflection-compression 50 \
    --reflection-period 10 \
    --reflection-disabled-for-crossover \
    --reflection-elitist-synchro \
    --evaluation-description-disabled \
    --ideas-coordinated-generation-disabled \
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