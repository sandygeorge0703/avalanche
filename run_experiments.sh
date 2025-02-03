#!/bin/bash

# Path to your Python script
SCRIPT_PATH="/gpfs01/home/egysg4/Documents/avalanche/train_LATEST.py"

# Loop to run the script 10 times and save results to different folders
for i in $(seq -f "%02g" 1 10)  # This will format the numbers as 01, 02, ..., 10
do
    # Set the experiment folder name
    EXPERIMENT_FOLDER="experiment_$i"

    # Ensure the folder exists
    mkdir -p "$EXPERIMENT_FOLDER"  # Create the folder if it doesn't exist

    # Run the Python script and pass the experiment folder as an argument
    python3 "$SCRIPT_PATH" --experiment_folder "$EXPERIMENT_FOLDER" > "$EXPERIMENT_FOLDER/output_experiment_$i.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: train_LATEST.py failed for experiment $i"
        exit 1
    fi

    echo "Run $i completed, results saved in $EXPERIMENT_FOLDER/output_experiment_$i.log"
done

# After all experiments, generate and save the final graph (losses plot)
echo "Generating the final graph of losses..."
