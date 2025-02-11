#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm  
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
import os
import torch
from avalanche.models import SimpleCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import EWC
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import nc_benchmark
from models.cnn_models import SimpleCNN



# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # If using a GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


# ## Define filepaths as constant

# In[ ]:


# Define file paths as constants
CSV_FILE_PATH = r'C:\Users\Sandhra George\avalanche\data\dataset.csv'
ROOT_DIR_PATH = r'C:\Users\Sandhra George\avalanche\caxton_dataset\print24'

csv_file = r'C:\Users\Sandhra George\avalanche\data\dataset.csv'  # Path to the CSV file
root_dir = r'C:\Users\Sandhra George\avalanche\caxton_dataset\print24'  # Path to the image directory


# ## Load data into DataFrame and filter print24

# In[ ]:


# Load data into a DataFrame for easier processing
data = pd.read_csv(CSV_FILE_PATH)

# Limit dataset to the images between row indices 454 and 7058 (inclusive)
#data_limited = data.iloc[454:7059].reset_index(drop=True)

# Filter the dataset to only include images containing "print24"
data_filtered = data[data.iloc[:, 0].str.contains('print24', na=False)]

# Update the first column to contain only the image filenames
data_filtered.iloc[:, 0] = data_filtered.iloc[:, 0].str.replace(r'.*?/(image-\d+\.jpg)', r'\1', regex=True)

# Display the updated DataFrame
print("First rows of filtered DataFrame:")
print(data_filtered.head())

# Display the last few rows of the updated DataFrame
print("\nLast rows of filtered DataFrame:")
print(data_filtered.tail())


# ## Analysing the target hotend temperature column

# In[ ]:


# Extract unique temperatures in the 'target_hotend' column and sort them
unique_temperatures = sorted(data_filtered['target_hotend'].unique())  # Sort temperatures in ascending order

# Calculate the full range of temperatures (min and max)
temperature_min = data_filtered['target_hotend'].min()
temperature_max = data_filtered['target_hotend'].max()

# Print the unique temperatures (sorted), count, and full range
print("\nUnique target hotend temperatures in the dataset (sorted):")
print(unique_temperatures)
print(f"\nNumber of unique target hotend temperatures: {len(unique_temperatures)}")
print(f"Temperature range: {temperature_min} to {temperature_max}")


# ## Create a random temperature sub list and new dataframes with equal class distribution

# In[ ]:


# Extract unique temperatures and sort them
unique_temperatures = sorted(data_filtered['target_hotend'].unique())  # Sort temperatures in ascending order

# Check if we have enough unique temperatures to select from
if len(unique_temperatures) >= 50:
    # Select the lowest and highest temperatures
    temperature_min = unique_temperatures[0]
    temperature_max = unique_temperatures[-1]

    # Remove the lowest and highest temperatures from the unique temperatures list
    remaining_temperatures = [temp for temp in unique_temperatures if temp != temperature_min and temp != temperature_max]

    # Randomly select 40 other temperatures from the remaining ones
    random_temperatures = random.sample(remaining_temperatures, 40)

    # Add the random temperatures to the temperature_sublist
    temperature_sublist = [temperature_min, temperature_max] + random_temperatures
    
    # Sort from lowest to highest hotend temperature
    temperature_sublist = sorted(temperature_sublist)

    # Print the temperature sublist
    print("\nTemperature sublist:")
    print(temperature_sublist)
    
    # Split into three experience groups
    split_size = len(temperature_sublist) // 3
    experience_1 = temperature_sublist[:split_size]  # First third
    experience_2 = temperature_sublist[split_size:2*split_size]  # Second third
    experience_3 = temperature_sublist[2*split_size:]  # Last third

    # Print the results
    print("\nExperience Group 1:", experience_1)
    print("\nExperience Group 2:", experience_2)
    print("\nExperience Group 3:", experience_3)
else:
    print("Not enough unique temperatures to select from. At least 50 unique temperatures are required.")
    experience_1 = experience_2 = experience_3 = []

# Initialize a dictionary to store DataFrames for each class per experience
experience_datasets = {1: {}, 2: {}, 3: {}}

# Iterate through the three experience groups
for exp_id, experience_temps in enumerate([experience_1, experience_2, experience_3], start=1):
    if not experience_temps:
        print(f"Skipping Experience {exp_id} due to insufficient temperatures.")
        continue

    print(f"\nProcessing Experience {exp_id} with temperatures: {experience_temps}...")

    # Filter the dataset based on the current experience's temperature range
    exp_data = data_filtered[data_filtered['target_hotend'].isin(experience_temps)]
    
    # Check if exp_data is empty after filtering
    if exp_data.empty:
        print(f"No data found for Experience {exp_id} with temperatures {experience_temps}. Skipping...")
        continue

    # Create a dictionary to store class-wise data for this experience
    class_datasets = {}

    # Iterate through each class (0, 1, 2) and filter data
    for class_id in [0, 1, 2]:
        class_data = exp_data[exp_data['hotend_class'] == class_id]
        
        if class_data.empty:
            print(f"Warning: Class {class_id} in Experience {exp_id} has no data!")
        else:
            class_datasets[class_id] = class_data
            print(f"Class {class_id} dataset size in Experience {exp_id}: {len(class_data)}")

    # Ensure that all classes have data before proceeding to balance
    if len(class_datasets) != 3:
        print(f"Skipping Experience {exp_id} because one or more classes are missing data!")
        continue  # Skip processing this experience if any class has no data

    # Find the smallest class size in this experience
    min_class_size = min(len(class_datasets[class_id]) for class_id in class_datasets)
    print(f"Smallest class size in Experience {exp_id}: {min_class_size}")

    # Balance the dataset for this experience
    balanced_data = []

    for class_id in class_datasets:
        class_data = class_datasets[class_id]
        # Randomly sample 'min_class_size' images from the class data to balance class distribution
        sampled_class_data = class_data.sample(n=min_class_size, random_state=42)  # Sample equally
        balanced_data.append(sampled_class_data)

    # Combine all class data for this experience into one balanced dataset
    balanced_dataset = pd.concat(balanced_data).reset_index(drop=True)

    # Shuffle the final balanced dataset
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Store the balanced dataset in the experience_datasets dictionary
    experience_datasets[exp_id] = balanced_dataset

    # Print summary for this experience
    print(f"\nBalanced dataset size for Experience {exp_id}: {len(balanced_dataset)}")
    print("Number of images in each class after balancing:")

    for class_id in [0, 1, 2]:
        class_count = len(balanced_dataset[balanced_dataset['hotend_class'] == class_id])
        print(f"Class {class_id}: {class_count} images")

    print("-" * 50)

# Print the first few rows for verification
for exp_id in [1, 2, 3]:
    if exp_id in experience_datasets:
        print(f"\nFirst five rows of Experience {exp_id} dataset:")
        print(experience_datasets[exp_id].head())


# ## Checking the class distribution of all the experience datasets

# In[ ]:


# Iterate over all experience datasets (1, 2, 3)
for exp_id in [1, 2, 3]:
    # Check if the experience dataset exists (in case an experience was skipped)
    if exp_id in experience_datasets:
        # Select only the 'img_path' and 'hotend_class' columns
        balanced_dataset_filtered = experience_datasets[exp_id][['img_path', 'hotend_class']]

        # Check the class distribution in the filtered dataset
        class_distribution = balanced_dataset_filtered['hotend_class'].value_counts()
        
        # Print the class distribution for the current experience
        print(f"\nClass distribution for Experience {exp_id}:")
        print(class_distribution)


# ## Printing the indices, the classes, and the number of images in each class

# In[ ]:


# Iterate over all experience datasets (1, 2, 3)
for exp_id in [1, 2, 3]:
    # Check if the experience dataset exists (in case an experience was skipped)
    if exp_id in experience_datasets:
        # Select only the 'img_path' and 'hotend_class' columns for the current experience dataset
        balanced_dataset_filtered = experience_datasets[exp_id][['img_path', 'hotend_class']]

        # Get the class distribution for the current experience dataset
        class_distribution = balanced_dataset_filtered['hotend_class'].value_counts()
        
        # Step 1: Print the indices, the classes, and the number of images in each class
        print(f"\n--- Experience {exp_id} ---")
        for class_label in class_distribution.index:
            # Get all indices for the current class
            class_indices = balanced_dataset_filtered[balanced_dataset_filtered['hotend_class'] == class_label].index.tolist()

            # Count the number of images for the current class
            num_images_in_class = len(class_indices)

            # Print the details for this class
            print(f"\nClass: {class_label} (Total images: {num_images_in_class})")
            print("Indices: ", class_indices)
            print(f"Number of images in class {class_label}: {num_images_in_class}")

        # Step 2: Get the number of unique classes
        num_classes = len(class_distribution)

        # Step 3: Set a small batch size
        small_batch_size = 15  # You can change this to a value like 32, 64, etc.

        # Step 4: Calculate the number of samples per class per batch
        samples_per_class = small_batch_size // num_classes  # Ensure it's divisible

        # Make sure we don't ask for more samples than available in the smallest class
        samples_per_class = min(samples_per_class, class_distribution.min())

        # Step 5: Calculate the total batch size
        batch_size = samples_per_class * num_classes

        print(f"\nRecommended Small Batch Size for Experience {exp_id}: {batch_size}")
        print(f"Samples per class in Experience {exp_id}: {samples_per_class}")
        print("-" * 50)  # To separate each experience's results


# ## At this point a balanced dataset for each experience has been created

# ## Create training, validation, and testing datasets

# In[ ]:


# Iterate over all experience datasets (1, 2, 3)
for exp_id in [1, 2, 3]:
    # Check if the experience dataset exists (in case an experience was skipped)
    if exp_id in experience_datasets:
        # Select only the 'img_path' and 'hotend_class' columns for the current experience dataset
        balanced_dataset_filtered = experience_datasets[exp_id][['img_path', 'hotend_class']]

        # Number of images per class (this will be the same after balancing)
        num_images_per_class = len(balanced_dataset_filtered) // 3  # Assuming there are 3 classes (0, 1, 2)

        # Calculate the number of samples per class for train, validation, and test sets
        train_size = int(0.8 * num_images_per_class)
        valid_size = int(0.1 * num_images_per_class)
        test_size = num_images_per_class - train_size - valid_size

        # Lists to hold indices for each class's dataset (train, validation, test)
        train_indices, valid_indices, test_indices = [], [], []

        # Split the data by class (assuming classes are 0, 1, 2)
        for class_label in [0, 1, 2]:
            class_data = balanced_dataset_filtered[balanced_dataset_filtered['hotend_class'] == class_label].index.tolist()

            # Shuffle the indices of the current class
            random.shuffle(class_data)

            # Split the indices for each class into train, validation, and test
            train_indices.extend(class_data[:train_size])
            valid_indices.extend(class_data[train_size:train_size + valid_size])
            test_indices.extend(class_data[train_size + valid_size:])

        # Sort the indices to ensure consistent processing
        train_indices, valid_indices, test_indices = sorted(train_indices), sorted(valid_indices), sorted(test_indices)

        # Create DataFrames for train, validation, and test sets based on the indices
        globals()[f'train_{exp_id}'] = balanced_dataset_filtered.loc[train_indices].reset_index(drop=True)
        globals()[f'valid_{exp_id}'] = balanced_dataset_filtered.loc[valid_indices].reset_index(drop=True)
        globals()[f'test_{exp_id}'] = balanced_dataset_filtered.loc[test_indices].reset_index(drop=True)

        # Count class distribution for each of the datasets
        def count_class_distribution(indices):
            class_counts = [0, 0, 0]  # Assuming 3 classes (0, 1, 2)
            for index in indices:
                class_label = balanced_dataset_filtered.loc[index, 'hotend_class']
                class_counts[class_label] += 1
            return class_counts

        # Count class distribution for each of the datasets
        train_class_distribution = count_class_distribution(train_indices)
        valid_class_distribution = count_class_distribution(valid_indices)
        test_class_distribution = count_class_distribution(test_indices)

        # Print the class distribution and dataset sizes
        print(f"\n--- Experience {exp_id} ---")
        print(f"Train set size: {len(train_indices)} | Class distribution: {train_class_distribution}")
        print(f"Validation set size: {len(valid_indices)} | Class distribution: {valid_class_distribution}")
        print(f"Test set size: {len(test_indices)} | Class distribution: {test_class_distribution}")

        print(f"Experience {exp_id} datasets created successfully!\n")

# Now, the datasets are directly available as:
# train_1, valid_1, test_1, train_2, valid_2, test_2, train_3, valid_3, test_3


# ## Check for Missing or Invalid Labels in Training, Validation, and Test Data

# In[ ]:


# Check for any missing labels or invalid labels
print(train_1['hotend_class'].isnull().sum())  # Count missing labels
print(train_1['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(train_2['hotend_class'].isnull().sum())  # Count missing labels
print(train_2['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(train_3['hotend_class'].isnull().sum())  # Count missing labels
print(train_3['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(valid_1['hotend_class'].isnull().sum())  # Count missing labels
print(valid_1['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(valid_2['hotend_class'].isnull().sum())  # Count missing labels
print(valid_2['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(valid_3['hotend_class'].isnull().sum())  # Count missing labels
print(valid_3['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(test_1['hotend_class'].isnull().sum())  # Count missing labels
print(test_1['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(test_2['hotend_class'].isnull().sum())  # Count missing labels
print(test_2['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

print(test_3['hotend_class'].isnull().sum())  # Count missing labels
print(test_3['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values


# ## Balanced Dataset class

# In[ ]:


# Define the dataset class
class BalancedDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validate that the images exist in the directory
        self.valid_indices = self.get_valid_indices()

    def get_valid_indices(self):
        valid_indices = []
        for idx in tqdm(range(len(self.data)), desc="Validating images"):
            img_name = self.data.iloc[idx, 0].strip()
            img_name = img_name.split('/')[-1]  # Extract file name
            
            if img_name.startswith("image-"):
                try:
                    # Ensure we only include images in the valid range
                    image_number = int(img_name.split('-')[1].split('.')[0])
                    if 4 <= image_number <= 26637:
                        full_img_path = os.path.join(self.root_dir, img_name)
                        if os.path.exists(full_img_path):
                            valid_indices.append(idx)
                        else:
                            print(f"Image does not exist: {full_img_path}")
                except ValueError:
                    print(f"Invalid filename format for {img_name}. Skipping...")
        
        print(f"Total valid indices found: {len(valid_indices)}")  # Debugging output
        return valid_indices

    def __len__(self):
            return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Wrap around the index if it exceeds the length of valid indices
        idx = idx % len(self.valid_indices)
        
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]
        img_name = self.data.iloc[actual_idx, 0].strip()
        full_img_path = os.path.join(self.root_dir, img_name)
    
        try:
            # Attempt to open the image and convert to RGB
            image = Image.open(full_img_path).convert('RGB')
    
            # Fetch the label and convert it to an integer
            label_str = self.data.iloc[actual_idx]['hotend_class']  # Use column name 'hotend_class'
            label = int(label_str)  # Ensure label is integer
    
            # Apply transformations if defined
            if self.transform:
                image = self.transform(image)
    
            return image, label, actual_idx
        except (OSError, IOError, ValueError) as e:
            # Print error message for debugging
            print(f"Error loading image {full_img_path}: {e}")
    
            # Handle gracefully by skipping the corrupted/missing file
            # Fetch the next valid index (recursively handle until a valid image is found)
            return self.__getitem__((idx + 1) % len(self.valid_indices))


# ## Balanced Batch Sampler class

# In[ ]:


class BalancedBatchSampler(Sampler):
    def __init__(self, data_frame, batch_size=15, samples_per_class=5):
        """
        data_frame: Pandas DataFrame with image paths and their respective class labels.
        batch_size: Total batch size.
        samples_per_class: Number of samples to draw from each class per batch.
        """
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_classes = len(data_frame['hotend_class'].unique())
        
        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")

        self.class_indices = {
            class_id: self.data_frame[self.data_frame['hotend_class'] == class_id].index.tolist()
            for class_id in self.data_frame['hotend_class'].unique()
        }
        
        # Shuffle class indices initially
        for class_id in self.class_indices:
            random.shuffle(self.class_indices[class_id])

        self.num_samples_per_epoch = sum(len(indices) for indices in self.class_indices.values())
        self.indices_used = {class_id: [] for class_id in self.class_indices}

    def __iter__(self):
        batches = []

        # Replenish indices for each class
        for class_id in self.class_indices:
            if not self.class_indices[class_id]:
                raise ValueError(f"Class {class_id} has no samples. Cannot form balanced batches.")

            # Shuffle and use all indices from this class
            self.indices_used[class_id] = self.class_indices[class_id].copy()
            random.shuffle(self.indices_used[class_id])

        # Generate balanced batches
        while len(batches) * self.batch_size < self.num_samples_per_epoch:
            batch = []
            for class_id in self.indices_used:
                if len(self.indices_used[class_id]) < self.samples_per_class:
                    # If a class runs out of samples, reshuffle and replenish
                    self.indices_used[class_id] = self.class_indices[class_id].copy()
                    random.shuffle(self.indices_used[class_id])

                # Take `samples_per_class` indices from the current class
                batch.extend(self.indices_used[class_id][:self.samples_per_class])
                self.indices_used[class_id] = self.indices_used[class_id][self.samples_per_class:]

            # Shuffle the batch and append
            random.shuffle(batch)
            batches.append(batch)

        return iter(batches)

    def __len__(self):
        # Total number of batches per epoch
        return self.num_samples_per_epoch // self.batch_size


# In[ ]:


# Define a dictionary to store datasets and DataLoaders
datasets = {}
dataloaders = {}

# Iterate over all experience datasets (1, 2, 3)
for exp_id in [1, 2, 3]:
    # Ensure the dataset exists
    if f"train_{exp_id}" in globals():
        train_data = globals()[f"train_{exp_id}"]
        val_data = globals()[f"valid_{exp_id}"]
        test_data = globals()[f"test_{exp_id}"]

        # Create dataset instances
        datasets[f"train_{exp_id}"] = BalancedDataset(data_frame=train_data, root_dir=root_dir)
        datasets[f"valid_{exp_id}"] = BalancedDataset(data_frame=val_data, root_dir=root_dir)
        datasets[f"test_{exp_id}"] = BalancedDataset(data_frame=test_data, root_dir=root_dir)

        # Create batch samplers for balanced training
        train_sampler = BalancedBatchSampler(data_frame=train_data, batch_size=15, samples_per_class=5)
        val_sampler = BalancedBatchSampler(data_frame=val_data, batch_size=15, samples_per_class=5)
        test_sampler = BalancedBatchSampler(data_frame=test_data, batch_size=15, samples_per_class=5)

        # Create DataLoaders
        dataloaders[f"train_{exp_id}"] = DataLoader(datasets[f"train_{exp_id}"], batch_sampler=train_sampler, shuffle=False)
        dataloaders[f"valid_{exp_id}"] = DataLoader(datasets[f"valid_{exp_id}"], batch_sampler=val_sampler, shuffle=False)
        dataloaders[f"test_{exp_id}"] = DataLoader(datasets[f"test_{exp_id}"], batch_sampler=test_sampler)

        # Print dataset lengths
        print(f"   Experience {exp_id} datasets and DataLoaders created successfully!")
        print(f"   Train dataset length: {len(datasets[f'train_{exp_id}'])}")
        print(f"   Validation dataset length: {len(datasets[f'valid_{exp_id}'])}")
        print(f"   Test dataset length: {len(datasets[f'test_{exp_id}'])}")


# ## Check the class distribution of randomly selected batches in train loader (for each experience)

# In[ ]:


# Function to print random batches and their class distribution for a given experience group
def print_random_batches(exp_id, num_batches=5):
    # Ensure the experience group exists
    if f"train_{exp_id}" not in dataloaders:
        print(f"Experience {exp_id} train loader not found!")
        return

    train_loader = dataloaders[f"train_{exp_id}"]  # Get the correct DataLoader

    print(f"\n  Random Batches for Experience {exp_id}  ")

    for _ in range(num_batches):
        # Get the next batch from the loader
        batch_images, batch_labels, _ = next(iter(train_loader))  # Get images, labels, and indices

        # Calculate the class distribution in this batch
        class_distribution = Counter(batch_labels.tolist())  # Convert tensor to list for counting

        # Print details
        print(f"\nBatch Class Distribution: {dict(class_distribution)}")
        print(f"Actual Labels: {batch_labels.tolist()}")
        print(f"Image Tensor Shape: {batch_images.shape}")
        print(f"Min Pixel Value: {batch_images.min()}, Max Pixel Value: {batch_images.max()}")
        print("-" * 50)

# Example Usage:
# Print batches for Experience 1
print_random_batches(exp_id=1, num_batches=5)

# Print batches for Experience 2
print_random_batches(exp_id=2, num_batches=5)

# Print batches for Experience 3
print_random_batches(exp_id=3, num_batches=5)


# ## Check class distribution of random batches from training, validation and testing data

# In[ ]:


def print_label_batch_from_loader(exp_id, dataset_type):
    """
    Fetch and print a batch of labels from the data loader for a specific experience group.

    Args:
        exp_id (int): The experience group number (1, 2, or 3).
        dataset_type (str): The dataset type - 'train', 'valid', or 'test'.
    """
    loader_key = f"{dataset_type}_{exp_id}"  # Example: 'train_1', 'valid_2', 'test_3'

    # Ensure the requested DataLoader exists
    if loader_key not in dataloaders:
        print(f"DataLoader for {loader_key} not found!")
        return

    loader = dataloaders[loader_key]  # Get the correct DataLoader

    # Fetch one batch
    data_iter = iter(loader)
    batch_images, batch_labels, _ = next(data_iter)  # Extract batch data

    # Print batch label details
    print(f"\nExperience {exp_id} - {dataset_type.capitalize()} Set - Sample Label Batch:")
    print(f"Labels Tensor: {batch_labels}")
    print(f"Labels as List: {batch_labels.tolist()}")
    print("-" * 50)

# Example Usage:
print_label_batch_from_loader(exp_id=1, dataset_type='train')  # Training batch from Experience 1
print_label_batch_from_loader(exp_id=2, dataset_type='valid')  # Validation batch from Experience 2
print_label_batch_from_loader(exp_id=3, dataset_type='test')   # Test batch from Experience 3


# ## Setting up a new folder for each experiment

# In[ ]:


# Set base directory
base_dir = "experiments"
os.makedirs(base_dir, exist_ok=True)

# Function to get the next experiment folder
def get_experiment_folder(exp_num):
    return os.path.join(base_dir, f"Experiment_{exp_num:02d}")  # Keeps two-digit format (01, 02, ..., 10)

# Set initial experiment number
experiment_num = 1
experiment_folder = get_experiment_folder(experiment_num)

# Create the main experiment directory if it doesn't exist
os.makedirs(experiment_folder, exist_ok=True)

# Set model path inside experiment folder
model_path = os.path.join(experiment_folder, "best_model.pth")

# Create subdirectories for training, validation, and test confusion matrices
train_folder = os.path.join(experiment_folder, "training_confusion_matrices")
val_folder = os.path.join(experiment_folder, "validation_confusion_matrices")
test_folder = os.path.join(experiment_folder, "test_confusion_matrices")

# Ensure that the subdirectories exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Print the directory where results will be saved
print(f"Saving results to: {experiment_folder}")


# ## Display a Random Image from the Dataset with Its Label

# In[ ]:


import random
import os
import matplotlib.pyplot as plt

def save_random_image_from_experiment(exp_id, dataset_type):
    """
    Selects a random image from the specified dataset (train, valid, or test) for a given experience ID,
    loads it, displays it, and saves it to the corresponding experiment folder.

    Args:
        exp_id (int): The experience group number (1, 2, or 3).
        dataset_type (str): The dataset type - 'train', 'valid', or 'test'.
    """
    # Ensure the dataset exists
    dataset_key = f"{dataset_type}_{exp_id}"  # Example: 'train_1', 'valid_2', 'test_3'
    if dataset_key not in datasets:
        print(f"Dataset {dataset_key} not found!")
        return

    dataset = datasets[dataset_key]  # Retrieve the dataset
    data_frame = dataset.data  # Get the underlying DataFrame

    # Ensure the dataset is not empty
    if data_frame.empty:
        print(f"Dataset {dataset_key} is empty!")
        return

    # Select a random index
    random_index = random.choice(data_frame.index)
    img_path = os.path.join(root_dir, data_frame.iloc[random_index, 0].strip())
    label = data_frame.loc[random_index, 'hotend_class']

    # Load and display the image
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Label: {label}")

    # Define the path to save the image inside the current experiment folder
    experiment_folder = os.path.join("experiments", f"experiment_{exp_id}")
    os.makedirs(experiment_folder, exist_ok=True)  # Ensure folder exists

    output_path = os.path.join(experiment_folder, f"random_{dataset_type}.png")

    # Save the figure
    plt.savefig(output_path)
    plt.clf()  # Clear the plot to avoid overlaps

    print(f"Image saved to: {output_path}")

# Example Usage:
save_random_image_from_experiment(exp_id=1, dataset_type='train')  # Random training image from Experience 1
save_random_image_from_experiment(exp_id=2, dataset_type='valid')  # Random validation image from Experience 2
save_random_image_from_experiment(exp_id=3, dataset_type='test')   # Random test image from Experience 3


# In[ ]:


# Iterate over all experience groups
for exp_id in [1, 2, 3]:  
    dataset_key = f"train_{exp_id}"  # e.g., 'train_1', 'train_2', 'train_3'
    
    # Ensure the dataset exists
    if dataset_key in datasets:
        data_frame = datasets[dataset_key].data  # Access the DataFrame from BalancedDataset

        # Ensure the dataset is not empty
        if not data_frame.empty:
            # First image
            first_index = data_frame.index[0]
            first_image = data_frame.loc[first_index, 'img_path']
            first_label = data_frame.loc[first_index, 'hotend_class']
            print(f"Experience {exp_id} - First Image Path: {first_image}, First Label: {first_label}")

            # Last image
            last_index = data_frame.index[-1]
            last_image = data_frame.loc[last_index, 'img_path']
            last_label = data_frame.loc[last_index, 'hotend_class']
            print(f"Experience {exp_id} - Last Image Path: {last_image}, Last Label: {last_label}\n")
        else:
            print(f"Experience {exp_id} - Training dataset is empty!\n")
    else:
        print(f"Experience {exp_id} - Training dataset not found!\n")


# # Creating the datasets for the EWC Strategy

# ## Implementing a Continual Learning Strategy (EWC) using Avalanche

# In[ ]:


class EWCCompatibleBalancedDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame with at least two columns: 'image_path' and 'hotend_class'
            root_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data_frame  # Stores the dataframe
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Assuming 'hotend_class' is the target column and 'image_path' is the column for image paths
        if 'hotend_class' in self.data.columns:
            self.targets = torch.tensor(self.data['hotend_class'].values)  # Labels from the 'hotend_class' column
        else:
            raise ValueError("DataFrame must have a column 'hotend_class' for labels.")

        if 'image_path' not in self.data.columns:
            raise ValueError("DataFrame must have a column 'image_path' to reference image files.")

        # Validate that the images exist in the directory
        self.valid_indices = self.get_valid_indices()

    def get_valid_indices(self):
        """Validates image paths and ensures they exist."""
        valid_indices = []
        for idx in tqdm(range(len(self.data)), desc="Validating images"):
            img_name = self.data.iloc[idx, 0].strip()  # Assuming first column is image path
            img_name = img_name.split('/')[-1]  # Extract file name
            
            if img_name.startswith("image-"):
                try:
                    # Ensure we only include images in the valid range
                    image_number = int(img_name.split('-')[1].split('.')[0])
                    if 4 <= image_number <= 26637:
                        full_img_path = os.path.join(self.root_dir, img_name)
                        if os.path.exists(full_img_path):
                            valid_indices.append(idx)
                        else:
                            print(f"Image does not exist: {full_img_path}")
                except ValueError:
                    print(f"Invalid filename format for {img_name}. Skipping...")
        
        print(f"Total valid indices found: {len(valid_indices)}")  # Debugging output
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Wrap around the index if it exceeds the length of valid indices
        idx = idx % len(self.valid_indices)
        
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]
        img_name = self.data.iloc[actual_idx, 0].strip()
        full_img_path = os.path.join(self.root_dir, img_name)
        label = self.targets[actual_idx]  # Get the label from the targets tensor
    
        try:
            # Attempt to open the image and convert to RGB
            image = Image.open(full_img_path).convert('RGB')
    
            # Apply transformations if defined
            if self.transform:
                image = self.transform(image)
    
            return image, label
        except (OSError, IOError, ValueError) as e:
            # Print error message for debugging
            print(f"Error loading image {full_img_path}: {e}")
    
            # Handle gracefully by skipping the corrupted/missing file
            return self.__getitem__((idx + 1) % len(self.valid_indices))  # Try next valid index

    def store_model_parameters(self, model):
        """Store the model's parameters after training a task"""
        self.model_parameters = {name: param.clone() for name, param in model.named_parameters()}
    
    def compute_fisher_information(self, model, dataloader, criterion):
        """Compute Fisher Information for EWC loss calculation"""
        fisher_information = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        
        model.eval()  # Set the model to evaluation mode
        for images, labels in dataloader:
            images, labels = images.to(model.device), labels.to(model.device)
            
            model.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            
            # Calculate Fisher Information for each parameter
            for name, param in model.named_parameters():
                fisher_information[name] += param.grad ** 2 / len(dataloader)
        
        # Store Fisher Information
        self.fisher_information = fisher_information
        return fisher_information
    
    def compute_ewc_loss(self, model, lambda_ewc=1000):
        """Compute the EWC loss based on stored parameters and Fisher Information"""
        ewc_loss = 0
        if hasattr(self, 'model_parameters') and hasattr(self, 'fisher_information'):
            for name, param in model.named_parameters():
                if name in self.model_parameters:
                    param_diff = param - self.model_parameters[name]
                    ewc_loss += (self.fisher_information[name] * param_diff ** 2).sum()
        
        return lambda_ewc * ewc_loss


# ## Creating training, validation and testing datasets to implement EWC

# In[ ]:


# Define the transformation (e.g., normalization)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Experience 1
train_data_exp1 = train_1.rename(columns={'img_path': 'image_path'})
val_data_exp1 = valid_1.rename(columns={'img_path': 'image_path'})
test_data_exp1 = test_1.rename(columns={'img_path': 'image_path'})

# Wrap the datasets using EWCCompatibleBalancedDataset
train_dataset_exp1 = EWCCompatibleBalancedDataset(data_frame=train_data_exp1, root_dir=root_dir, transform=transform)
val_dataset_exp1 = EWCCompatibleBalancedDataset(data_frame=val_data_exp1, root_dir=root_dir, transform=transform)
test_dataset_exp1 = EWCCompatibleBalancedDataset(data_frame=test_data_exp1, root_dir=root_dir, transform=transform)

# Create the BalancedBatchSampler instances for Experience 1
train_sampler_exp1 = BalancedBatchSampler(data_frame=train_data_exp1, batch_size=15, samples_per_class=5)
val_sampler_exp1 = BalancedBatchSampler(data_frame=val_data_exp1, batch_size=15, samples_per_class=5)
test_sampler_exp1 = BalancedBatchSampler(data_frame=test_data_exp1, batch_size=15, samples_per_class=5)

# Create DataLoaders for Experience 1
train_loader1 = DataLoader(train_dataset_exp1, batch_sampler=train_sampler_exp1, shuffle=False)
val_loader1 = DataLoader(val_dataset_exp1, batch_sampler=val_sampler_exp1, shuffle=False)
test_loader1 = DataLoader(test_dataset_exp1, batch_sampler=test_sampler_exp1, shuffle=False)

# Optionally print a message indicating successful creation of the DataLoader for Experience 1
print("Experience 1 DataLoaders created successfully!")

# Experience 2
train_data_exp2 = train_2.rename(columns={'img_path': 'image_path'})
val_data_exp2 = valid_2.rename(columns={'img_path': 'image_path'})
test_data_exp2 = test_2.rename(columns={'img_path': 'image_path'})

# Wrap the datasets using EWCCompatibleBalancedDataset
train_dataset_exp2 = EWCCompatibleBalancedDataset(data_frame=train_data_exp2, root_dir=root_dir, transform=transform)
val_dataset_exp2 = EWCCompatibleBalancedDataset(data_frame=val_data_exp2, root_dir=root_dir, transform=transform)
test_dataset_exp2 = EWCCompatibleBalancedDataset(data_frame=test_data_exp2, root_dir=root_dir, transform=transform)

# Create the BalancedBatchSampler instances for Experience 2
train_sampler_exp2 = BalancedBatchSampler(data_frame=train_data_exp2, batch_size=15, samples_per_class=5)
val_sampler_exp2 = BalancedBatchSampler(data_frame=val_data_exp2, batch_size=15, samples_per_class=5)
test_sampler_exp2 = BalancedBatchSampler(data_frame=test_data_exp2, batch_size=15, samples_per_class=5)

# Create DataLoaders for Experience 2
train_loader2 = DataLoader(train_dataset_exp2, batch_sampler=train_sampler_exp2, shuffle=False)
val_loader2 = DataLoader(val_dataset_exp2, batch_sampler=val_sampler_exp2, shuffle=False)
test_loader2 = DataLoader(test_dataset_exp2, batch_sampler=test_sampler_exp2, shuffle=False)

# Optionally print a message indicating successful creation of the DataLoader for Experience 2
print("Experience 2 DataLoaders created successfully!")

# Experience 3
train_data_exp3 = train_3.rename(columns={'img_path': 'image_path'})
val_data_exp3 = valid_3.rename(columns={'img_path': 'image_path'})
test_data_exp3 = test_3.rename(columns={'img_path': 'image_path'})

# Wrap the datasets using EWCCompatibleBalancedDataset
train_dataset_exp3 = EWCCompatibleBalancedDataset(data_frame=train_data_exp3, root_dir=root_dir, transform=transform)
val_dataset_exp3 = EWCCompatibleBalancedDataset(data_frame=val_data_exp3, root_dir=root_dir, transform=transform)
test_dataset_exp3 = EWCCompatibleBalancedDataset(data_frame=test_data_exp3, root_dir=root_dir, transform=transform)

# Create the BalancedBatchSampler instances for Experience 3
train_sampler_exp3 = BalancedBatchSampler(data_frame=train_data_exp3, batch_size=15, samples_per_class=5)
val_sampler_exp3 = BalancedBatchSampler(data_frame=val_data_exp3, batch_size=15, samples_per_class=5)
test_sampler_exp3 = BalancedBatchSampler(data_frame=test_data_exp3, batch_size=15, samples_per_class=5)

# Create DataLoaders for Experience 3
train_loader3 = DataLoader(train_dataset_exp3, batch_sampler=train_sampler_exp3, shuffle=False)
val_loader3 = DataLoader(val_dataset_exp3, batch_sampler=val_sampler_exp3, shuffle=False)
test_loader3 = DataLoader(test_dataset_exp3, batch_sampler=test_sampler_exp3, shuffle=False)

# Optionally print a message indicating successful creation of the DataLoader for Experience 3
print("Experience 3 DataLoaders created successfully!")


# In[ ]:





# ## Checking the class distribution of random batches from each experience group

# In[ ]:


# Function to print random batches and their class distribution
def print_random_batches(train_loader1, num_batches=5):
    for _ in range(num_batches):
        # Get the next batch from the loader
        batch_images, batch_labels = next(iter(train_loader1))  # Get the images, labels, and indices (if needed)
        
        # Calculate the class distribution in this batch
        class_distribution = Counter(batch_labels.tolist())  # Convert tensor to list for counting
        
        # Print the class distribution for the current batch
        print(f"Class distribution for this batch: {dict(class_distribution)}")
        
        # Print the actual labels for the batch (as a list or tensor)
        print("Actual labels for this batch:")
        print(batch_labels.tolist())  # Converts tensor to list for readability
        
        # Print the image tensor shape for the batch
        print("Image tensor shape for the batch:")
        print(batch_images.shape)  # This prints the shape of the image tensor
        
        # Optionally, print a few details of the image tensors (e.g., min and max values) to understand them
        print("Min and max values of the image tensors:")
        print(f"Min: {batch_images.min()}, Max: {batch_images.max()}")
        
        # If you want to print the image itself (assuming it's a small size, for visualization)
        # You can use something like matplotlib to visualize the images, for example:
        # from matplotlib import pyplot as plt
        # plt.imshow(batch_images[0].permute(1, 2, 0).numpy())  # assuming 3 channel images
        # plt.show()
        
        print("-" * 50)

# Print random batches and their class distribution
print_random_batches(train_loader1, num_batches=5)
print_random_batches(val_loader1, num_batches=5)
print_random_batches(test_loader1, num_batches=5)


# ## Checking datasets attributes

# In[ ]:


# Check if the dataset has the 'data' and 'targets' attributes
def check_dataset(train_dataset_exp1):
    # Check the 'data' (images) and 'targets' (labels) attributes
    try:
        print(f"Data attribute type: {type(train_dataset_exp1.data)}")
        print(f"Targets attribute type: {type(train_dataset_exp1.targets)}")
        
        # Print the shapes of the data and targets to confirm dimensions
        print(f"Shape of data (images): {train_dataset_exp1.data.shape}")
        print(f"Shape of targets (labels): {train_dataset_exp1.targets.shape}")
        
        # Check the type and a few elements from the dataset
        print("Sample data and target:", train_dataset_exp1[0])  # Check the first sample (image, label)
    except AttributeError as e:
        print(f"Error accessing dataset attributes: {e}")

# Running the function on my dataset
check_dataset(train_dataset_exp1)


# ## Debugging statements

# In[ ]:


print(f"Total training samples: {len(train_dataset_exp1)}")
print(f"Total test samples: {len(test_dataset_exp1)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")


# In[ ]:


# Debugging function to inspect class distribution across the datasets
from collections import Counter

def print_class_distribution(dataset_list, dataset_name, sample_size=1000):
    """Efficiently prints dataset class distributions using a sample."""
    print(f"\nChecking {dataset_name} Dataset Class Distributions...\n")

    for i, dataset in enumerate(dataset_list):
        labels = []
        
        # Sample a subset of data for efficiency
        sample_indices = range(min(sample_size, len(dataset)))  # Limit to `sample_size`
        
        for idx in sample_indices:
            labels.append(int(dataset[idx][1].item()))  # Convert tensor label to int
        
        # Count occurrences of each class
        label_counts = Counter(labels)
        print(f"{dataset_name} Dataset {i}: Unique Classes -> {set(label_counts.keys())}, Counts -> {dict(label_counts)}")

# Print class distributions before creating the benchmark
print_class_distribution([train_dataset_exp1, train_dataset_exp2, train_dataset_exp3], "Train")
print_class_distribution([test_dataset_exp1, test_dataset_exp2, test_dataset_exp3], "Test")


# ## Check if the datasets are in the correct format for using Avalanche

# In[ ]:


# Check if your dataset is a subclass of torch.utils.data.Dataset
isinstance(train_dataset_exp1, torch.utils.data.Dataset)  # This should return True


# In[ ]:


len(train_dataset_exp1)  # This should return the total number of samples in the dataset.


# In[ ]:


image, label = train_dataset_exp1[0]  # Try accessing the first element
print(image.shape, label)  # This should return a tensor (image) and an integer (label)


# In[ ]:


image, label = train_dataset_exp1[0]  # Access the first sample
print(f"Image shape: {image.shape}, Label: {label}")


# In[ ]:


from torch.utils.data import DataLoader

# Create DataLoader for your dataset
train_loader = DataLoader(train_dataset_exp1, batch_size=15, shuffle=True)

# Iterate through the DataLoader and check batches
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}: Images shape {images.shape}, Labels: {labels}")
    break  # Check the first batch only


# In[ ]:


print("Classes in train_dataset_exp1:", set(train_dataset_exp1.targets.numpy()))
print("Classes in train_dataset_exp2:", set(train_dataset_exp2.targets.numpy()))
print("Classes in train_dataset_exp3:", set(train_dataset_exp3.targets.numpy()))
print(train_dataset_exp1)


# In[ ]:


from avalanche.benchmarks import ni_benchmark
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

# Create a simple benchmark using your dataset
benchmark = ni_benchmark(
    train_dataset=[train_dataset_exp1, train_dataset_exp2, train_dataset_exp3],
    test_dataset=[test_dataset_exp1, test_dataset_exp2, test_dataset_exp3],
    n_experiences=3,
)

# Print the benchmark experiences
for experience in benchmark.train_stream:
    print(f"Start of experience: {experience.current_experience}")
    print(f"Classes in this experience: {experience.classes_in_this_experience}")


# ## Implementing EWC using Avalanche

# In[ ]:


##### THIS IS THE VERSION I AM CURRENTLY USING ########
from avalanche.benchmarks import nc_benchmark
from tqdm import tqdm
import time
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, forgetting_metrics, StreamConfusionMatrix
from torch.utils.data import DataLoader
from avalanche.training import EWC
from avalanche.benchmarks import ni_benchmark

# Generate the benchmark with explicit fixed experience assignment
benchmark = ni_benchmark(
    train_dataset=[train_dataset_exp1, train_dataset_exp2, train_dataset_exp3],  # List of training datasets
    test_dataset=[test_dataset_exp1, test_dataset_exp2, test_dataset_exp3],  # List of test datasets
    n_experiences=3,  # 3 experiences, one for each dataset
    task_labels=[0,0,0],  # We don't need separate task labels in domain-incremental learning
    shuffle=False,  # No shuffling to maintain domain consistency across experiences
    seed=1234,  # Reproducibility seed
    balance_experiences=True,  # Optional: Ensure balanced experiences if needed
)

# Print the classes in each experience to verify correctness
for experience in benchmark.train_stream:
    print(f"Start of experience {experience.current_experience}")
    print(f"Classes in this experience: {experience.classes_in_this_experience}")


# Model creation
model = SimpleCNN(num_classes=3).to(device)

# Define the evaluation plugin and loggers
tb_logger = TensorboardLogger()
text_logger = TextLogger(open('log.txt', 'a'))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=3, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

# Create the strategy instance (EWC)
cl_strategy = EWC(
    model=model,  
    optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),  
    criterion=CrossEntropyLoss(),  
    train_mb_size=4,
    train_epochs=1,
    eval_mb_size=16,
    ewc_lambda=0.4,
    evaluator=eval_plugin
)

# Starting experiment...
print("Starting experiment...")
results = []

# Run the training strategy with your corrected benchmark
for experience in benchmark.train_stream:
    print(f"Start of experience: {experience.current_experience}")
    print(f"Current Classes: {experience.classes_in_this_experience}")
    
    # Ensure `start_time` is initialized before timing
    start_time = time.time() 

    # Check if the experience is correctly loaded
    print(f"Experience {experience.current_experience}: {len(experience.dataset)} samples")

    # Create DataLoader for the current experience
    if experience.current_experience == 0:
        train_loader = train_loader1
        test_loader = test_loader1
    elif experience.current_experience == 1:
        train_loader = train_loader2
        test_loader = test_loader2
    else:
        train_loader = train_loader3
        test_loader = test_loader3

    # Print batch sizes and input/output shapes in the train_loader
    print(f"Number of batches in this experience: {len(train_loader)}")
    
    for i, (train_batch_data, train_batch_labels) in enumerate(train_loader):
        print(f"Batch {i}: Input shape = {train_batch_data.shape}, Labels shape = {train_batch_labels.shape}")
        if i > 5:  # Check for the first 6 batches to avoid printing too much
            break
    
    # Get number of batches for progress bar
    num_batches = len(train_loader)
    
    for epoch in range(cl_strategy.train_epochs):  # Manually loop over the epochs
        print(f"Starting epoch {epoch+1}/{cl_strategy.train_epochs}...")

        with tqdm(total=num_batches, desc=f"Training Experience {experience.current_experience}") as pbar:
            for i, (train_batch_data, train_batch_labels) in enumerate(train_loader):
                # Ensure data is on the same device as the model
                train_batch_data, train_batch_labels = train_batch_data.to(device), train_batch_labels.to(device)
                
                # Print the batch shapes at each iteration for further debugging
                print(f"Training Batch {i}: x shape = {train_batch_data.shape}, y shape = {train_batch_labels.shape}")
                
                # Pack into a batch (data, labels)
                train_batch = (train_batch_data, train_batch_labels)
    
                # Train the model on the minibatch
                try:
                    res = cl_strategy.train(experience, num_workers=0)
                except Exception as e:
                    print(f"Error during training in experience {experience.current_experience}, batch {i}: {e}")
                    break
    
                # Update progress bar
                pbar.update(1)

        # Log epoch-level metrics at the end of each epoch
        print(f"End of epoch {epoch+1}: "
              f"Loss: {res['Loss_Epoch/train_phase/train_stream/Task000']:.4f}, "
              f"Accuracy: {res['Top1_Acc_Epoch/train_phase/train_stream/Task000']:.2f}%")

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    # Evaluate the model after training the experience
    print("Evaluating after training...")
    start_eval_time = time.time()
    
    # Ensure we pass the correct experience object, not a DataLoader
    with tqdm(total=1, desc=f"Evaluating Experience {experience.current_experience}") as eval_pbar:
        try:
            res_eval = cl_strategy.eval([experience])  # Pass the experience, not test_loader
        except Exception as e:
            print(f"Error during evaluation in experience {experience.current_experience}: {e}")
            continue
        eval_pbar.update(1)  # Update progress bar after evaluation
    
    eval_time = time.time() - start_eval_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"Evaluation results: {res_eval}")

    results.append(res_eval)
    print("-" * 50)  # Add separator between experiences


# In[ ]:




