#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="/gpfs01/home/egysg4/Documents/avalanche/increased_data/increased_data_benchmark_model.py"

# Check if the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH" >&2
    exit 1
fi

# Loop to run the script 5 times
for i in {1..5}; do
    # Create a unique results folder for this run
    RUN_DIR="run_$i"
    mkdir -p "$RUN_DIR"

    # Define the log file path
    LOG_FILE="$RUN_DIR/run.log"

    echo "Starting run $i. Logs will be saved to $LOG_FILE"

    # Run your Python script and redirect both stdout and stderr to the log file
    python "$SCRIPT_PATH" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    # Check if the Python script ran successfully
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Error: Python script encountered an error during execution on run $i (exit code: $EXIT_CODE)" >&2
    else
        echo "Run $i completed successfully."
    fi
done

echo "All runs completed."
