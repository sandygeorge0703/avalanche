#!/bin/bash

# Ensure MinGW-w64 binaries are in the PATH (if needed for your environment)
export PATH="/mingw64/bin:$PATH"

# Path to your Python scripts
SCRIPT_PATH="/gpfs01/home/egysg4/Documents/avalanche/train_LATEST.py"
PROCESS_LOGS_PATH="/gpfs01/home/egysg4/Documents/avalanche/process_logs.py"

# Create a folder for the final output (this will be used to store the final plot)
mkdir -p "$HOME/final_output"

# Loop to run the script 10 times and save results to different folders
for i in {2..10}
do
    # Create a directory for each run (e.g., experiment_1, experiment_2, ..., experiment_10)
    FOLDER="experiment_$i"
    mkdir -p "$FOLDER"  # Create the directory if it doesn't exist

    # Run the Python script with python3 and save the output in the folder
    python3 "$SCRIPT_PATH" > "$FOLDER/output_experiment_$i.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: train_new.py failed for experiment $i"
        exit 1
    fi

    echo "Run $i completed, results saved in $FOLDER/output_experiment_$i.log"

    # Now, process the logs for this experiment by running the Python script
    python3 "$PROCESS_LOGS_PATH" "$i"
    if [ $? -ne 0 ]; then
        echo "Error: process_logs.py failed for experiment $i"
        exit 1
    fi
done

# After all experiments, generate and save the final graph (losses plot)
echo "Generating the final graph of losses..."