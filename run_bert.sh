#!/bin/bash

###############################################################################
# CONFIGURATION
###############################################################################
# Total number of target models to train
NUM_TARGET=500
# Fixed random seed for all target jobs
RANDOM_SEED_TARGET=522
# Define start and end indices for splitting the jobs
SEED_START=0
SEED_END=$SEED_START+$NUM_TARGET
# Define the chunk size (number of jobs to submit in each batch)
CHUNK_SIZE=15

ALPHA=0.4

# Create logs directory if it doesn't exist
mkdir -p ./logs_bert

###############################################################################
# FUNCTION: submit_target_job
# Submits a target model training job with 1 GPU on node g05 using RTX 6000.
# The job trains fine_tune_target.py with --random_seed RANDOM_SEED_TARGET and
# --split_seed provided as argument.
###############################################################################
submit_target_job() {
  local SPLIT_SEED=$1
  local job_out
  job_out=$(sbatch --gres=gpu:1 --mem=1000 --time=1:00:00 --chdir=$(pwd) --output=./logs_bert/target_model_${SPLIT_SEED}_%j.out --error=./logs_bert/target_model_${SPLIT_SEED}_%j.err <<EOF
#!/bin/bash
#SBATCH --job-name=target_${RANDOM_SEED_TARGET}_${SPLIT_SEED}
# Load necessary modules and activate the environment
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate common

echo "Training target model with random_seed ${RANDOM_SEED_TARGET} and split_seed ${SPLIT_SEED}"
python Fine-tune-bert.py --SGD_New --target_sup pos --alpha ${ALPHA} --random_seed ${RANDOM_SEED_TARGET} --split_seed ${SPLIT_SEED}
EOF
)
  # Extract and return the job ID
  echo $(echo $job_out | awk '{print $NF}')
}

###############################################################################
# MAIN LOGIC: Submit target model jobs in chunks and wait for each batch to finish.
###############################################################################
for ((chunk_start=SEED_START; chunk_start<=SEED_END; chunk_start+=CHUNK_SIZE)); do
  chunk_end=$((chunk_start + CHUNK_SIZE - 1))
  if [ $chunk_end -gt $SEED_END ]; then
    chunk_end=$SEED_END
  fi
  echo "Processing target model jobs for split seeds ${chunk_start} to ${chunk_end}"

  # Array to store job IDs for this chunk
  TARGET_JOB_IDS=()

  for SPLIT_SEED in $(seq $chunk_start $chunk_end); do
    echo "Submitting target model job for SPLIT_SEED=$SPLIT_SEED"
    jobid=$(submit_target_job $SPLIT_SEED)
    if [ -z "$jobid" ]; then
      echo "Error: Target job for SPLIT_SEED=$SPLIT_SEED failed to submit."
      continue
    else
      echo "Submitted target job for SPLIT_SEED=$SPLIT_SEED, jobid=$jobid"
      TARGET_JOB_IDS+=($jobid)
    fi
  done

  # Wait for all target jobs in this batch to finish.
  echo "Waiting for target model jobs for split seeds [${chunk_start}..${chunk_end}] to finish..."
  while true; do
    # Only output job names (-o "%j") and suppress the header (-h)
    pending_count=$(squeue -h -o "%j" -u $USER | grep -c "^target_${RANDOM_SEED_TARGET}_")
    if [ "$pending_count" -eq 0 ]; then
      break
    fi
    sleep 10
  done
  echo "All target model jobs for split seeds [${chunk_start}..${chunk_end}] completed."
done

python score_bert.py
