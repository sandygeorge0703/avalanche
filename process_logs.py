import re
import os
import matplotlib.pyplot as plt

# Base directory where all experiment folders are stored
base_dir = "/gpfs01/home/egysg4/Documents/avalanche"

# Number of experiments
num_experiments = 10

# Dictionary to hold losses for all experiments
all_losses = {}

# Loop through all experiment directories
for i in range(1, num_experiments + 1):
    # Path to the current log file
    log_file_path = os.path.join(base_dir, f"experiment_{i}", f"output_experiment_{i}.log")

    # Initialize lists for train and val losses for this experiment
    train_losses = []
    val_losses = []

    # Check if the log file exists
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                # Extract train losses
                train_matches = re.findall(r"train_losses:\s*([\d.]+)", line)
                train_losses.extend([float(value) for value in train_matches])

                # Extract val losses
                val_matches = re.findall(r"val_losses:\s*([\d.]+)", line)
                val_losses.extend([float(value) for value in val_matches])

        # Store the losses for this experiment
        all_losses[f"experiment_{i}"] = {"train_losses": train_losses, "val_losses": val_losses}
    else:
        print(f"Log file not found for experiment_{i}: {log_file_path}")

# Plot all train and val losses
plt.figure(figsize=(10, 6))
for experiment, losses in all_losses.items():
    # Plot train losses
    plt.plot(losses["train_losses"], label=f"{experiment} - Train Losses", linestyle="--")
    # Plot val losses
    plt.plot(losses["val_losses"], label=f"{experiment} - Val Losses", linestyle="-")

# Customize the plot
plt.title("Train and Validation Losses for All Experiments")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Save the plot to a new folder
output_folder = os.path.join(base_dir, "final_output")
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
output_path = os.path.join(output_folder, "losses_plot.png")
plt.savefig(output_path)

# Show confirmation and plot
print(f"Losses plot saved to: {output_path}")
plt.show()
