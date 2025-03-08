#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm  
import torch
from collections import Counter
from torch.utils.data import ConcatDataset
import random
import os
import torchvision.transforms as transforms
import pandas as pd

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ## Creating dataset for multiple parts

# In[ ]:

# Define file paths as constants for Linux HPC
CSV_FILE_PATH = '/gpfs01/home/egysg4/Documents/avalanche/data/dataset.csv'
ROOT_DIR_PATH = '/gpfs01/home/egysg4/Documents/avalanche/caxton_dataset'

# Load data into a DataFrame for easier processing
data = pd.read_csv(CSV_FILE_PATH)

# Filter the dataset to include images containing "print24", "print131", or "print0"
pattern = 'print24|print131|print0|print46|print86|print109'
data_filtered = data[data.iloc[:, 0].str.contains(pattern, na=False)]

# Update the first column to include both the print folder and the image filename.
# The regex now captures the folder name (print24, print131, or print0) and the image filename.
data_filtered.iloc[:, 0] = data_filtered.iloc[:, 0].str.replace(
    r'.*?/(print24|print131|print0|print46|print86|print109)/(image-\d+\.jpg)', 
    r'\1/\2', 
    regex=True
)

# Display the updated DataFrame
print("First rows of filtered DataFrame:")
print(data_filtered.head())

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
if len(unique_temperatures) >= 69:
    # Select the lowest and highest temperatures
    temperature_min = unique_temperatures[0]
    temperature_max = unique_temperatures[-1]

    # Remove the lowest and highest temperatures from the unique temperatures list
    remaining_temperatures = [temp for temp in unique_temperatures if temp != temperature_min and temp != temperature_max]

    # Randomly select 40 other temperatures from the remaining ones
    random_temperatures = random.sample(remaining_temperatures, 50)

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
        train_size = int(0.7 * num_images_per_class)
        valid_size = int(0.15 * num_images_per_class)
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


# ## BalancedBatchSampler class

# In[ ]:


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_frame, batch_size=15, samples_per_class=5):
        """
        data_frame: Pandas DataFrame with image paths and their respective class labels.
        batch_size: Total batch size.
        samples_per_class: Number of samples to draw from each class per batch.
        """
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # Get unique class labels and count them.
        unique_classes = data_frame['hotend_class'].unique()
        self.num_classes = len(unique_classes)
        if self.num_classes == 0:
            raise ValueError("No class labels found in the provided DataFrame. Unique classes: " + str(unique_classes))
        
        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")
        
        # Build a dictionary of indices per class.
        self.class_indices = {
            class_id: self.data_frame[self.data_frame['hotend_class'] == class_id].index.tolist()
            for class_id in unique_classes
        }
        for class_id in self.class_indices:
            random.shuffle(self.class_indices[class_id])
        self.num_samples_per_epoch = sum(len(indices) for indices in self.class_indices.values())
        self.indices_used = {class_id: [] for class_id in self.class_indices}
    
    def __iter__(self):
        batches = []
        # Replenish indices for each class.
        for class_id in self.class_indices:
            if not self.class_indices[class_id]:
                raise ValueError(f"Class {class_id} has no samples. Cannot form balanced batches.")
            self.indices_used[class_id] = self.class_indices[class_id].copy()
            random.shuffle(self.indices_used[class_id])
        # Generate balanced batches.
        while len(batches) * self.batch_size < self.num_samples_per_epoch:
            batch = []
            for class_id in self.indices_used:
                if len(self.indices_used[class_id]) < self.samples_per_class:
                    self.indices_used[class_id] = self.class_indices[class_id].copy()
                    random.shuffle(self.indices_used[class_id])
                batch.extend(self.indices_used[class_id][:self.samples_per_class])
                self.indices_used[class_id] = self.indices_used[class_id][self.samples_per_class:]
            random.shuffle(batch)
            batches.append(batch)
        return iter(batches)
    
    def __len__(self):
        return self.num_samples_per_epoch // self.batch_size

 


# ## BalancedDataset Class

# In[ ]:


class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, root_dir, transform=None, debug=False, max_retries=5):
        self.debug = debug
        self.root_dir = root_dir
        # Reset index to ensure proper positional indexing.
        self.data = data_frame.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.max_retries = max_retries
        if self.debug:
            print(f"Dataset length (filtered): {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        attempts = 0
        original_idx = idx
        while attempts < self.max_retries:
            idx = idx % len(self.data)
            try:
                # Use .iloc for positional indexing.
                row = self.data.iloc[idx]
                img_path = row.iloc[0].strip()  # e.g., "print24/image-123.jpg"
                full_img_path = os.path.join(self.root_dir, img_path)
                label = row.iloc[1]
                image = Image.open(full_img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                if self.debug:
                    print(f"Error loading image at index {idx} ({full_img_path}): {e}")
                # Try the next index.
                idx += 1
                attempts += 1
        # If max_retries reached without success, raise an error.
        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded for index {original_idx}.")


# ## Filter and Reindex Function

# In[ ]:


#####################################
# 3. Filter and Reindex Function
#####################################
def filter_and_reindex(data_frame, root_dir):
    """
    Filters the DataFrame to include only rows with valid image paths
    and then reindexes the DataFrame so that indices are contiguous.
    """
    valid_indices = []
    allowed_folders = {"print24", "print131", "print0", "print46","print86","print109"}
    for idx in range(len(data_frame)):
        img_path = data_frame.iloc[idx, 0].strip()
        parts = img_path.split('/')
        if len(parts) < 2:
            continue
        folder, file_name = parts[0], parts[1]
        if folder not in allowed_folders:
            continue
        if not file_name.startswith("image-"):
            continue
        full_img_path = os.path.join(root_dir, folder, file_name)
        if os.path.exists(full_img_path):
            valid_indices.append(idx)
    filtered_df = data_frame.iloc[valid_indices].reset_index(drop=True)
    return filtered_df


# ## Function to Print Random Batch Distributions

# In[ ]:


#####################################
# 4. Function to Print Random Batch Distributions
#####################################
def print_random_batches_from_loader(loader, num_batches=3, dataset_name="Dataset"):
    print(f"\nRandom batch label distributions for {dataset_name}:")
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        label_counts = Counter(labels.numpy())
        print(f"Batch {batch_idx + 1} distribution: {label_counts}")


# In[ ]:


#####################################
# 5. Main Code: Loop Over Experiments (1, 2, 3)
#####################################

# Define the root directory
ROOT_DIR_PATH = '/gpfs01/home/egysg4/Documents/avalanche/caxton_dataset'
root_dir = ROOT_DIR_PATH


# Loop over experiment numbers.
for exp_id in [1, 2, 3]:
    print(f"\n=== Processing Experience {exp_id} ===")
    # Access the corresponding global DataFrames.
    # Ensure these DataFrames (train_1, train_2, etc.) are defined.
    original_train_df = globals()[f"train_{exp_id}"]
    original_valid_df = globals()[f"valid_{exp_id}"]
    original_test_df  = globals()[f"test_{exp_id}"]

    # Filter and reindex for each split.
    filtered_train_data = filter_and_reindex(original_train_df, root_dir)
    filtered_valid_data = filter_and_reindex(original_valid_df, root_dir)
    filtered_test_data  = filter_and_reindex(original_test_df, root_dir)

    # Create dataset instances using the filtered DataFrames.
    train_dataset = BalancedDataset(filtered_train_data, root_dir, debug=False)
    valid_dataset = BalancedDataset(filtered_valid_data, root_dir, debug=False)
    test_dataset  = BalancedDataset(filtered_test_data, root_dir, debug=False)

    # Create balanced batch samplers using the same filtered DataFrames.
    train_sampler = BalancedBatchSampler(data_frame=filtered_train_data, batch_size=15, samples_per_class=5)
    valid_sampler = BalancedBatchSampler(data_frame=filtered_valid_data, batch_size=15, samples_per_class=5)
    test_sampler  = BalancedBatchSampler(data_frame=filtered_test_data,  batch_size=15, samples_per_class=5)

    # Create DataLoaders using the custom balanced batch samplers.
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler)
    test_loader  = DataLoader(test_dataset,  batch_sampler=test_sampler)

    # Print random batch distributions for each split.
    print_random_batches_from_loader(train_loader, num_batches=3, dataset_name=f"Experience {exp_id} Train")
    print_random_batches_from_loader(valid_loader, num_batches=3, dataset_name=f"Experience {exp_id} Validation")
    print_random_batches_from_loader(test_loader,  num_batches=3, dataset_name=f"Experience {exp_id} Test")


# ## Checking class distribution in each dataset

# In[ ]:


from collections import Counter

def count_class_distribution(df):
    """
    Count occurrences of each class in the 'hotend_class' column of the DataFrame.
    """
    return Counter(df['hotend_class'])

# Loop over all experiments (assuming they are named train_1, valid_1, test_1, etc.)
for exp_id in [1, 2, 3]:
    # Retrieve each dataset using globals()
    train_df = globals()[f"train_{exp_id}"]
    valid_df = globals()[f"valid_{exp_id}"]
    test_df  = globals()[f"test_{exp_id}"]
    
    # Count the class distribution
    train_dist = count_class_distribution(train_df)
    valid_dist = count_class_distribution(valid_df)
    test_dist  = count_class_distribution(test_df)
    
    # Print the results
    print(f"\n--- Experience {exp_id} ---")
    print(f"Train dataset size: {len(train_df)} | Class distribution: {train_dist}")
    print(f"Validation dataset size: {len(valid_df)} | Class distribution: {valid_dist}")
    print(f"Test dataset size: {len(test_df)} | Class distribution: {test_dist}")


# ## Creating experience datasets

# In[ ]:


from torch.utils.data import ConcatDataset

def wrap_dataset(ds, root_dir):
    """
    Given ds, which may be a BalancedDataset (with a .data attribute)
    or a plain DataFrame, return a new BalancedDataset constructed
    from the underlying DataFrame with a reset index.
    """
    if hasattr(ds, 'data'):
        df = ds.data.reset_index(drop=True)
    else:
        # Assume ds is a DataFrame.
        df = ds.reset_index(drop=True)
    return BalancedDataset(df, root_dir, debug=False)

# Experience 1 datasets (single datasets)
exp1_train = globals()["train_1"]
exp1_valid = globals()["valid_1"]
exp1_test  = globals()["test_1"]

# For Experience 1_2, re-wrap the underlying DataFrames and then concatenate.
exp1_2_train = ConcatDataset([
    wrap_dataset(globals()["train_1"], root_dir),
    wrap_dataset(globals()["train_2"], root_dir)
])
exp1_2_valid = ConcatDataset([
    wrap_dataset(globals()["valid_1"], root_dir),
    wrap_dataset(globals()["valid_2"], root_dir)
])
exp1_2_test = ConcatDataset([
    wrap_dataset(globals()["test_1"], root_dir),
    wrap_dataset(globals()["test_2"], root_dir)
])

# For Experience 1_2_3, re-wrap and concatenate datasets from experiences 1, 2, and 3.
exp1_2_3_train = ConcatDataset([
    wrap_dataset(globals()["train_1"], root_dir),
    wrap_dataset(globals()["train_2"], root_dir),
    wrap_dataset(globals()["train_3"], root_dir)
])
exp1_2_3_valid = ConcatDataset([
    wrap_dataset(globals()["valid_1"], root_dir),
    wrap_dataset(globals()["valid_2"], root_dir),
    wrap_dataset(globals()["valid_3"], root_dir)
])
exp1_2_3_test = ConcatDataset([
    wrap_dataset(globals()["test_1"], root_dir),
    wrap_dataset(globals()["test_2"], root_dir),
    wrap_dataset(globals()["test_3"], root_dir)
])


# In[ ]:


def create_balanced_loader(dataset, root_dir, batch_size=15, samples_per_class=5):
    """
    Given an experience dataset (plain DataFrame, BalancedDataset, or ConcatDataset),
    create a DataLoader using a BalancedBatchSampler built from the underlying data.
    """
    # If dataset is a plain DataFrame, wrap it.
    if isinstance(dataset, pd.DataFrame):
        dataset = BalancedDataset(dataset, root_dir, debug=False)
    
    # Determine the DataFrame to use for the sampler.
    if hasattr(dataset, 'data'):
        data_for_sampler = dataset.data.reset_index(drop=True)
    elif isinstance(dataset, ConcatDataset):
        data_frames = []
        for d in dataset.datasets:
            if hasattr(d, 'data'):
                data_frames.append(d.data.reset_index(drop=True))
            elif isinstance(d, pd.DataFrame):
                data_frames.append(d.reset_index(drop=True))
            else:
                raise ValueError("Sub-dataset type not recognized.")
        data_for_sampler = pd.concat(data_frames, ignore_index=True)
    else:
        raise ValueError("Dataset type not recognized for sampler creation.")
    
    sampler = BalancedBatchSampler(data_frame=data_for_sampler, batch_size=batch_size, samples_per_class=samples_per_class)
    loader = DataLoader(dataset, batch_sampler=sampler)
    return loader


# In[ ]:


# =============================================================================
# Create DataLoaders for each experience dataset.
# =============================================================================

exp1_train_loader    = create_balanced_loader(exp1_train, root_dir, batch_size=15, samples_per_class=5)
exp1_valid_loader    = create_balanced_loader(exp1_valid, root_dir, batch_size=15, samples_per_class=5)
exp1_test_loader     = create_balanced_loader(exp1_test,  root_dir, batch_size=15, samples_per_class=5)

exp1_2_train_loader  = create_balanced_loader(exp1_2_train, root_dir, batch_size=15, samples_per_class=5)
exp1_2_valid_loader  = create_balanced_loader(exp1_2_valid, root_dir, batch_size=15, samples_per_class=5)
exp1_2_test_loader   = create_balanced_loader(exp1_2_test,  root_dir, batch_size=15, samples_per_class=5)

exp1_2_3_train_loader = create_balanced_loader(exp1_2_3_train, root_dir, batch_size=15, samples_per_class=5)
exp1_2_3_valid_loader = create_balanced_loader(exp1_2_3_valid, root_dir, batch_size=15, samples_per_class=5)
exp1_2_3_test_loader  = create_balanced_loader(exp1_2_3_test,  root_dir, batch_size=15, samples_per_class=5)


# In[ ]:


from collections import Counter

def print_batches_from_loader(loader, num_batches=3, dataset_name="Dataset"):
    """
    Iterates through the given DataLoader and prints the label distribution
    for the first `num_batches` batches.
    
    Args:
        loader (DataLoader): The DataLoader to iterate over.
        num_batches (int): Number of batches to print.
        dataset_name (str): Name of the dataset (for printing purposes).
    """
    print(f"\nBatch label distributions for {dataset_name}:")
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        # Ensure labels are on CPU and convert to a numpy array for counting.
        label_counts = Counter(labels.cpu().numpy())
        print(f"Batch {batch_idx + 1} distribution: {label_counts}")


# In[ ]:


# For Experience 1:
print_batches_from_loader(exp1_train_loader, num_batches=3, dataset_name="Experience 1 Train")
print_batches_from_loader(exp1_valid_loader, num_batches=3, dataset_name="Experience 1 Validation")
print_batches_from_loader(exp1_test_loader,  num_batches=3, dataset_name="Experience 1 Test")

# For Experience 1_2:
print_batches_from_loader(exp1_2_train_loader, num_batches=3, dataset_name="Experience 1_2 Train")
print_batches_from_loader(exp1_2_valid_loader, num_batches=3, dataset_name="Experience 1_2 Validation")
print_batches_from_loader(exp1_2_test_loader,  num_batches=3, dataset_name="Experience 1_2 Test")

# For Experience 1_2_3:
print_batches_from_loader(exp1_2_3_train_loader, num_batches=3, dataset_name="Experience 1_2_3 Train")
print_batches_from_loader(exp1_2_3_valid_loader, num_batches=3, dataset_name="Experience 1_2_3 Validation")
print_batches_from_loader(exp1_2_3_test_loader,  num_batches=3, dataset_name="Experience 1_2_3 Test")


# ## Checking class distribution in each experience dataset

# In[ ]:


from collections import Counter
from torch.utils.data import ConcatDataset
import pandas as pd

def count_classes(dataset):
    counts = Counter()
    
    # If the dataset is a plain DataFrame:
    if isinstance(dataset, pd.DataFrame):
        values = dataset.iloc[:, 1].tolist()
        counts.update(values)
    # If it's a ConcatDataset:
    elif isinstance(dataset, ConcatDataset):
        for d in dataset.datasets:
            if isinstance(d, pd.DataFrame):
                values = d.iloc[:, 1].tolist()
            elif hasattr(d, 'data'):
                values = d.data.iloc[:, 1].tolist()
            else:
                raise ValueError("Sub-dataset type not recognized.")
            counts.update(values)
    # If it's an object with a 'data' attribute:
    elif hasattr(dataset, 'data'):
        values = dataset.data.iloc[:, 1].tolist()
        counts.update(values)
    else:
        raise ValueError("Dataset type not recognized.")
    
    return counts

# Now, print the class distributions for your experience datasets.
# For Experience 1:
print("Class distribution in Experience 1 train dataset:", count_classes(exp1_train))
print("Class distribution in Experience 1 valid dataset:", count_classes(exp1_valid))
print("Class distribution in Experience 1 test dataset:", count_classes(exp1_test))

# For Experience 1_2:
print("Class distribution in Experience 1_2 train dataset:", count_classes(exp1_2_train))
print("Class distribution in Experience 1_2 valid dataset:", count_classes(exp1_2_valid))
print("Class distribution in Experience 1_2 test dataset:", count_classes(exp1_2_test))

# For Experience 1_2_3:
print("Class distribution in Experience 1_2_3 train dataset:", count_classes(exp1_2_3_train))
print("Class distribution in Experience 1_2_3 valid dataset:", count_classes(exp1_2_3_valid))
print("Class distribution in Experience 1_2_3 test dataset:", count_classes(exp1_2_3_test))


# In[ ]:


def get_dataset_size(dataset):
    # If the dataset is a plain DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return len(dataset)
    # If it's a ConcatDataset:
    elif isinstance(dataset, ConcatDataset):
        return sum(len(d) for d in dataset.datasets)
    # If it has a __len__ attribute (like BalancedDataset)
    elif hasattr(dataset, '__len__'):
        return len(dataset)
    else:
        raise ValueError("Dataset type not recognized.")

# Now, print the sizes and class distributions for your experience datasets.
# For Experience 1:
print("Size of Experience 1 train dataset:", get_dataset_size(exp1_train))
print("Class distribution in Experience 1 train dataset:", count_classes(exp1_train))
print("Size of Experience 1 valid dataset:", get_dataset_size(exp1_valid))
print("Class distribution in Experience 1 valid dataset:", count_classes(exp1_valid))
print("Size of Experience 1 test dataset:", get_dataset_size(exp1_test))
print("Class distribution in Experience 1 test dataset:", count_classes(exp1_test))

# For Experience 1_2:
print("Size of Experience 1_2 train dataset:", get_dataset_size(exp1_2_train))
print("Class distribution in Experience 1_2 train dataset:", count_classes(exp1_2_train))
print("Size of Experience 1_2 valid dataset:", get_dataset_size(exp1_2_valid))
print("Class distribution in Experience 1_2 valid dataset:", count_classes(exp1_2_valid))
print("Size of Experience 1_2 test dataset:", get_dataset_size(exp1_2_test))
print("Class distribution in Experience 1_2 test dataset:", count_classes(exp1_2_test))

# For Experience 1_2_3:
print("Size of Experience 1_2_3 train dataset:", get_dataset_size(exp1_2_3_train))
print("Class distribution in Experience 1_2_3 train dataset:", count_classes(exp1_2_3_train))
print("Size of Experience 1_2_3 valid dataset:", get_dataset_size(exp1_2_3_valid))
print("Class distribution in Experience 1_2_3 valid dataset:", count_classes(exp1_2_3_valid))
print("Size of Experience 1_2_3 test dataset:", get_dataset_size(exp1_2_3_test))
print("Class distribution in Experience 1_2_3 test dataset:", count_classes(exp1_2_3_test))


# ## Creating experience dataloaders

# ## Benchmark experiment with training and validation accuracy plot

# In[ ]:

import sys
import os
# Add the parent directory of this script (which should contain the 'models' folder) to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn_models import SimpleCNN


import os
import csv
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix

# Define the experiment configurations in a dictionary.
experiments = {
    "experience_1": (exp1_train, exp1_valid, exp1_test),
    "experience_1_2": (exp1_2_train, exp1_2_valid, exp1_2_test),
    "experience_1_2_3": (exp1_2_3_train, exp1_2_3_valid, exp1_2_3_test)
}

# Create a mapping between experiment names and your new DataLoader variables.
# (Make sure these variables are already created: exp1_train_loader, exp1_valid_loader, etc.)
# For example:
#   #"experience_1" --> exp1_train_loader, exp1_valid_loader, exp1_test_loader
#   #"experience_1_2" --> exp1_2_train_loader, exp1_2_valid_loader, exp1_2_test_loader
#   "experience_1_2_3" --> exp1_2_3_train_loader, exp1_2_3_valid_loader, exp1_2_3_test_loader

# Get the current working directory and define the benchmark folder.
current_dir = os.getcwd()
benchmark_folder = os.path.join(current_dir, "benchmark_experiment_increased_data")
os.makedirs(benchmark_folder, exist_ok=True)
print(f"Benchmark folder created at: {benchmark_folder}")

# Training settings
num_epochs = 30
num_classes = 3  # update if needed

# Loop over each experiment configuration.
for exp_name, _ in experiments.items():
    print(f"\nStarting experiment: {exp_name}\n")
    
    # Create a subfolder for this experiment.
    exp_folder = os.path.join(benchmark_folder, exp_name)
    os.makedirs(exp_folder, exist_ok=True)
    
    # Set the best model path.
    best_model_path = os.path.join(exp_folder, f"model_{exp_name}.pth")
    
    # Retrieve the appropriate pre-created DataLoaders.
    if exp_name == "experience_1":
        train_loader = exp1_train_loader
        val_loader = exp1_valid_loader
        test_loader = exp1_test_loader
    elif exp_name == "experience_1_2":
        train_loader = exp1_2_train_loader
        val_loader = exp1_2_valid_loader
        test_loader = exp1_2_test_loader
    elif exp_name == "experience_1_2_3":
        train_loader = exp1_2_3_train_loader
        val_loader = exp1_2_3_valid_loader
        test_loader = exp1_2_3_test_loader
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")
    
    # Set device to GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model, loss function, optimizer, and scheduler.
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize confusion matrix trackers.
    train_cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    val_cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    
    # For plotting losses and accuracies.
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Create CSV file to store epoch losses and accuracies.
    csv_file_path = os.path.join(exp_folder, "training_validation_losses.csv")
    header = ["Epoch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    
    best_val_accuracy = 0.0
    start_epoch = 0  # always start fresh for each experiment
    
    # ----------------- Training and Validation Loop -----------------
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        class_counts = [0] * num_classes
        
        # Training phase with progress bar.
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            print(f"Outputs (Raw): {outputs}")  # Log raw outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update confusion matrix and class counts.
            train_cm.update(predicted, labels)
            for label in labels:
                class_counts[label.item()] += 1
                
            # Print predicted vs actual labels for each batch.
            for i in range(len(labels)):
                print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
        
        train_epoch_loss = running_loss / total_samples
        train_epoch_accuracy = correct_predictions / total_samples
        print(f"Training Loss: {train_epoch_loss:.4f}, Training Accuracy: {train_epoch_accuracy:.4f}")
        print(f"Training Class Distribution: {class_counts}")
        
        # Update learning rate scheduler.
        scheduler.step()
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        
        # Compute and save training confusion matrix.
        cm_train = train_cm.compute()
        print(f"Training Confusion Matrix:\n{cm_train}")
        sns.heatmap(cm_train.cpu().numpy(), annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Training Confusion Matrix - Epoch {epoch + 1}')
        output_path_train = os.path.join(exp_folder, f"training_confusion_matrix_epoch_{epoch + 1}.png")
        plt.savefig(output_path_train)
        plt.clf()  # Clear the plot
        print(f"Training Confusion Matrix saved to: {output_path_train}")
        train_cm.reset()
        
        # ----------------- Validation Phase -----------------
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        val_class_counts = [0] * num_classes
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                print(f"Outputs (Raw): {outputs}")  # Log raw outputs
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)
                
                val_cm.update(predicted, labels)
                for label in labels:
                    val_class_counts[label.item()] += 1
                
                for i in range(len(labels)):
                    print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
        
        val_epoch_loss = val_loss / val_total_samples
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")
        print(f"Validation Class Distribution: {val_class_counts}")
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        
        cm_val = val_cm.compute()
        print(f"Validation Confusion Matrix:\n{cm_val}")
        sns.heatmap(cm_val.cpu().numpy(), annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Validation Confusion Matrix - Epoch {epoch + 1}')
        output_path_val = os.path.join(exp_folder, f"validation_confusion_matrix_epoch_{epoch + 1}.png")
        plt.savefig(output_path_val)
        plt.clf()
        print(f"Validation Confusion Matrix saved to: {output_path_val}")
        val_cm.reset()
        
        # Save the best model if validation accuracy improves.
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_accuracy": best_val_accuracy
            }, best_model_path)
            print(f"Saved best model for {exp_name} at epoch {epoch + 1} with accuracy {best_val_accuracy:.4f}")
        
        # Append losses and accuracies to CSV.
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy])
    
    print(f"Experiment {exp_name} training complete. Losses saved to: {csv_file_path}")
    
    # ----------------- Testing Phase -----------------
    def test_model(model, test_loader):
        model.eval()
        correct_predictions = 0
        total_samples = 0
        test_class_counts = [0] * num_classes
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                for label in labels:
                    test_class_counts[label.item()] += 1
                for i in range(len(labels)):
                    print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
        avg_accuracy = correct_predictions / total_samples
        print(f"Test Accuracy: {avg_accuracy:.4f}")
        print(f"Test Class Distribution: {test_class_counts}")
        
        from sklearn.metrics import confusion_matrix
        cm_test = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
        print(f"Test Confusion Matrix:\n{cm_test}")
        
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Test Confusion Matrix')
        output_path_test = os.path.join(exp_folder, "test_confusion_matrix.png")
        plt.savefig(output_path_test)
        plt.clf()
        print(f"Test Confusion Matrix saved to: {output_path_test}")
    
    test_model(model, test_loader)
    
    # Plot the training and validation losses.
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses for {exp_name}')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(exp_folder, "training_validation_loss.png")
    plt.savefig(loss_plot_path)
    plt.clf()
    print(f"Training and Validation Loss plot saved to: {loss_plot_path}")
    
    # Plot the training and validation accuracies.
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracies for {exp_name}')
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = os.path.join(exp_folder, "training_validation_accuracy.png")
    plt.savefig(accuracy_plot_path)
    plt.clf()
    print(f"Training and Validation Accuracy plot saved to: {accuracy_plot_path}")

print("\nAll benchmark experiments completed.")





