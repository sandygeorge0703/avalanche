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


# Define file paths as constants
CSV_FILE_PATH = r'C:\Users\Sandhra George\avalanche\data\dataset.csv'
ROOT_DIR_PATH = r'C:\Users\Sandhra George\avalanche\caxton_dataset'  # Common parent directory
root_dir = ROOT_DIR_PATH

# Load data into a DataFrame for easier processing
data = pd.read_csv(CSV_FILE_PATH)

# Filter the dataset to include images containing "print24", "print131", or "print0"
pattern = 'print24|print131|print0|print46|print82|print111|print132|print122|print37'
data_filtered = data[data.iloc[:, 0].str.contains(pattern, na=False)]

# Update the first column to include both the print folder and the image filename.
# The regex now captures the folder name (print24, print131, or print0) and the image filename.
data_filtered.iloc[:, 0] = data_filtered.iloc[:, 0].str.replace(
    r'.*?/(print24|print131|print0|print46|print82|print111|print132|print122|print37)/(image-\d+\.jpg)', 
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


import random
import pandas as pd

unique_temperatures = sorted(data_filtered['target_hotend'].unique())

if len(unique_temperatures) >= 60:
    temperature_min = unique_temperatures[0]
    temperature_max = unique_temperatures[-1]
    remaining_temperatures = [temp for temp in unique_temperatures if temp not in [temperature_min, temperature_max]]
    random_temperatures = random.sample(remaining_temperatures, 51)
    temperature_sublist = sorted([temperature_min, temperature_max] + random_temperatures)
    
    # Split the temperature sublist into three groups (roughly equal thirds)
    split_size = len(temperature_sublist) // 3
    experience_1 = temperature_sublist[:split_size]
    experience_2 = temperature_sublist[split_size:2*split_size]
    experience_3 = temperature_sublist[2*split_size:]
    
    print("Temperature sublist:", temperature_sublist)
    print("\nExperience Group 1:", experience_1)
    print("Experience Group 2:", experience_2)
    print("Experience Group 3:", experience_3)
else:
    print("Not enough unique temperatures to select from.")
    experience_1 = experience_2 = experience_3 = []

# Create a dictionary to store datasets for each experience (non-cumulative)
experience_datasets = {}

# Assign each experience a specific class: 
# Experience 1 -> class 0, Experience 2 -> class 1, Experience 3 -> class 2
for exp_id, (experience_temps, target_class) in enumerate(zip([experience_1, experience_2, experience_3], [0, 1, 2]), start=1):
    if not experience_temps:
        print(f"Skipping Experience {exp_id} due to insufficient temperatures.")
        continue
    print(f"\nProcessing Experience {exp_id} for class {target_class} with temperatures: {experience_temps}...")
    
    # Print initial class distribution for the experience's temperatures (before filtering by target_class)
    exp_data_all = data_filtered[data_filtered['target_hotend'].isin(experience_temps)]
    print("Initial class distribution in Experience", exp_id, ":")
    for class_id in [0, 1, 2]:
        class_count = len(exp_data_all[exp_data_all['hotend_class'] == class_id])
        print(f"Class {class_id}: {class_count}")
    
    # Filter data for the current experience's temperatures and the target class
    exp_data = exp_data_all[exp_data_all['hotend_class'] == target_class]
    
    if exp_data.empty:
        print(f"No data found for Experience {exp_id} with class {target_class}. Skipping...")
        continue

    # Ensure each experience dataset has exactly x images.
    # Enforce a fixed size for each experience:
    desired_size = 20000
    if len(exp_data) >= desired_size:
        exp_data = exp_data.sample(n=desired_size, random_state=42)
    else:
        exp_data = exp_data.sample(n=desired_size, replace=True, random_state=42)
    experience_datasets[exp_id] = exp_data
    print(f"Dataset size for Experience {exp_id} (class {target_class}): {len(exp_data)}")


# In[ ]:


import random
import pandas as pd

# Define split proportions
train_prop = 0.7
valid_prop = 0.2
# test_prop is computed as remainder

def stratified_split(df, train_prop=0.7, valid_prop=0.2, random_state=42):
    """
    Splits a DataFrame into stratified train, validation, and test sets based on 'hotend_class'.
    """
    train_list, valid_list, test_list = [], [], []
    
    # Group the DataFrame by the class column
    for cls, group in df.groupby('hotend_class'):
        group_shuffled = group.sample(frac=1, random_state=random_state)
        n = len(group_shuffled)
        n_train = int(train_prop * n)
        n_valid = int(valid_prop * n)
        
        train_list.append(group_shuffled.iloc[:n_train])
        valid_list.append(group_shuffled.iloc[n_train:n_train+n_valid])
        test_list.append(group_shuffled.iloc[n_train+n_valid:])
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    valid_df = pd.concat(valid_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    return train_df, valid_df, test_df

# Combine experiences 1 and 2 into one dataset: experience_1_2
combined_dataset = pd.DataFrame()
for exp_id in [1, 2]:
    if exp_id in experience_datasets:
        # Use only the necessary columns from each experience
        data = experience_datasets[exp_id][['img_path', 'hotend_class']]
        combined_dataset = pd.concat([combined_dataset, data], ignore_index=True)

# Use stratified splitting for Experience 1_2
if not combined_dataset.empty:
    experience_1_2_train, experience_1_2_valid, experience_1_2_test = stratified_split(combined_dataset, train_prop, valid_prop)
    
    total_images = len(combined_dataset)
    print("\n--- Experience 1_2 Stratified Splits ---")
    print(f"Total images in Experience 1_2: {total_images}")
    print(f"Train set size: {len(experience_1_2_train)}")
    print(f"Validation set size: {len(experience_1_2_valid)}")
    print(f"Test set size: {len(experience_1_2_test)}")
    print("Class distribution (combined):", combined_dataset['hotend_class'].value_counts().to_dict())

# Process experience 3 separately using the original non-stratified approach
if 3 in experience_datasets:
    dataset = experience_datasets[3][['img_path', 'hotend_class']]
    indices = dataset.index.tolist()
    random.shuffle(indices)
    
    total_images = len(indices)
    train_count = int(train_prop * total_images)
    valid_count = int(valid_prop * total_images)
    # The test set gets the remaining images
    test_count = total_images - train_count - valid_count
    
    experience_3_train = dataset.loc[indices[:train_count]].reset_index(drop=True)
    experience_3_valid = dataset.loc[indices[train_count:train_count + valid_count]].reset_index(drop=True)
    experience_3_test  = dataset.loc[indices[train_count + valid_count:]].reset_index(drop=True)
    
    print("\n--- Experience 3 Splits ---")
    print(f"Total images in Experience 3: {total_images}")
    print(f"Train set size: {len(experience_3_train)}")
    print(f"Validation set size: {len(experience_3_valid)}")
    print(f"Test set size: {len(experience_3_test)}")
    print("Class distribution (Experience 3):", dataset['hotend_class'].value_counts().to_dict())


# ## BalancedBatchSamplerClass

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
        self.num_classes = len(data_frame['hotend_class'].unique())
        
        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")

        # Build a dictionary of indices per class.
        self.class_indices = {
            class_id: self.data_frame[self.data_frame['hotend_class'] == class_id].index.tolist()
            for class_id in self.data_frame['hotend_class'].unique()
        }
        for class_id in self.class_indices:
            random.shuffle(self.class_indices[class_id])
        self.num_samples_per_epoch = sum(len(indices) for indices in self.class_indices.values())
        self.indices_used = {class_id: [] for class_id in self.class_indices}
    
    def __iter__(self):
        indices_used = {cid: self.class_indices[cid].copy() for cid in self.class_indices}
        for indices in indices_used.values():
            random.shuffle(indices)
        
        num_batches = min(len(indices) for indices in indices_used.values()) // self.samples_per_class
        batches = []
        for b in range(num_batches):
            print(f"Before batch {b+1}, indices available per class:")
            for cid in indices_used:
                print(f"  Class {cid}: {len(indices_used[cid])} indices left")
            batch = []
            for cid in self.class_indices:
                batch.extend(indices_used[cid][:self.samples_per_class])
                indices_used[cid] = indices_used[cid][self.samples_per_class:]
            random.shuffle(batch)
            batches.append(batch)
        return iter(batches)


# You can define __len__ to be a fixed number of batches per epoch if needed.


    def __len__(self):
        return min(len(indices) for indices in self.class_indices.values()) // self.samples_per_class    


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
        # Use .iloc for positional indexing.
        row = self.data.iloc[idx]
        img_path = row.iloc[0].strip()  # e.g., "print24/image-123.jpg"
        full_img_path = os.path.join(self.root_dir, img_path)
        label = row.iloc[1]
        try:
            image = Image.open(full_img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            if self.debug:
                print(f"Error loading image at index {idx} ({full_img_path}): {e}")
            # Instead of shifting the index, sample a replacement from the same class.
            same_class_df = self.data[self.data.iloc[:, 1] == label]
            if same_class_df.empty:
                raise RuntimeError(f"No replacement available for class {label}.")
            replacement_idx = random.choice(same_class_df.index.tolist())
            # Try loading the replacement image.
            row = self.data.iloc[replacement_idx]
            img_path = row.iloc[0].strip()
            full_img_path = os.path.join(self.root_dir, img_path)
            try:
                image = Image.open(full_img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                raise RuntimeError(f"Failed to load replacement image for index {idx} (class {label}): {e}")


# ## Filter and reindex function

# In[ ]:


def filter_and_reindex(data_frame, root_dir):
    """
    Filters the DataFrame to include only rows with valid image paths
    and then reindexes the DataFrame so that indices are contiguous.
    """
    valid_indices = []
    allowed_folders = {"print24", "print131", "print0", "print46","print82","print111","print132", "print37","print122"}
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


# ## Creating a Naive Class which inherits from AvalancheDataset and contains all the expected functions

# In[ ]:


import os
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute
from avalanche.benchmarks.utils.transforms import TupleTransform

class NaiveCompatibleBalancedDataset(AvalancheDataset):
    def __init__(self, data_frame, root_dir=None, transform=None, task_label=0, indices=None):
        """
        Custom dataset compatible with Naive that inherits from AvalancheDataset.
        It loads images from disk, applies transforms, and provides sample-wise
        attributes for targets and task labels.
        
        Args:
            data_frame (pd.DataFrame or list): If a DataFrame, it must contain columns
                'image_path' and 'hotend_class'. If a list, it is assumed to be a pre-built
                list of datasets (used in subset calls).
            root_dir (str, optional): Directory where images are stored. Must be provided if data_frame is a DataFrame.
            transform (callable, optional): Transformations to apply.
            task_label (int, optional): Task label for continual learning.
            indices (Sequence[int], optional): Optional indices for subsetting.
        """
        # If data_frame is a list, assume this is a call from subset() and forward the call.
        if isinstance(data_frame, list):
            super().__init__(data_frame, indices=indices)
            return

        # Otherwise, data_frame is a DataFrame. Ensure root_dir is provided.
        if root_dir is None:
            raise ValueError("root_dir must be provided when data_frame is a DataFrame")
        
        # Reset DataFrame index for consistency.
        self.data = data_frame.reset_index(drop=True)
        self.root_dir = root_dir
        self.task_label = task_label

        # Define a default transform if none provided.
        default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Wrap the transform in TupleTransform so that it applies only to the image element.
        self._transform_groups = {
            "train": TupleTransform([transform or default_transform]),
            "eval": TupleTransform([transform or default_transform])
        }
        
        # Ensure required columns exist.
        if 'hotend_class' not in self.data.columns:
            raise ValueError("DataFrame must contain 'hotend_class' for labels.")
        if 'image_path' not in self.data.columns:
            raise ValueError("DataFrame must contain 'image_path' for image paths.")
        
        # Validate image paths and obtain valid indices.
        valid_indices = self.get_valid_indices()
        if len(valid_indices) == 0:
            raise ValueError("No valid image paths found.")
        
        # Compute targets and task labels for valid samples.
        targets_data = torch.tensor(self.data.loc[valid_indices, 'hotend_class'].values)
        targets_task_labels_data = torch.full_like(targets_data, self.task_label)
        
        # Prepare sample entries (one per valid image).
        samples = []
        for idx in valid_indices:
            img_name = self.data.loc[idx, 'image_path'].strip()
            full_img_path = os.path.join(self.root_dir, img_name)
            label = int(self.data.loc[idx, 'hotend_class'])
            samples.append({
                "img_path": full_img_path,
                "label": label,
                "task_label": self.task_label
            })
        
        # Define an internal basic dataset that loads images.
        class BasicDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                img_path = sample["img_path"]
                try:
                    # Load the image (ensure it is a PIL image).
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # If an error occurs, try the next sample.
                    return self.__getitem__((idx + 1) % len(self.samples))
                return image, sample["label"], sample["task_label"]
        
        basic_dataset = BasicDataset(samples)
        
        # Create data attributes.
        data_attributes = [
            DataAttribute(targets_data, name="targets", use_in_getitem=True),
            DataAttribute(targets_task_labels_data, name="targets_task_labels", use_in_getitem=True)
        ]
        
        # IMPORTANT: Pass the basic_dataset inside a list so that AvalancheDataset
        # correctly sets up its internal flat data, and forward the indices parameter.
        super().__init__(
            [basic_dataset],
            data_attributes=data_attributes,
            transform_groups=self._transform_groups,
            indices=indices
        )
    
    def get_valid_indices(self):
        """Return indices for which the image file exists."""
        valid_indices = []
        for idx in tqdm(range(len(self.data)), desc="Validating images"):
            img_name = self.data.loc[idx, 'image_path'].strip()
            full_img_path = os.path.join(self.root_dir, img_name)
            if os.path.exists(full_img_path):
                valid_indices.append(idx)
            else:
                print(f"Image does not exist: {full_img_path}")
        print(f"Total valid images: {len(valid_indices)}")
        return valid_indices


# ## Creating training, validation and testing datasets to implement EWC

# In[ ]:


from torchvision import transforms
import pandas as pd

# Define the transformation (e.g., normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Combine Experience 1 and 2
# ---------------------------
# Use the already split DataFrames from the combined dataset.
# Here we filter and reindex these DataFrames and rename 'img_path' to 'image_path'.

filtered_train_data_exp1_2 = filter_and_reindex(experience_1_2_train, root_dir).rename(
    columns={'img_path': 'image_path'}
)
filtered_valid_data_exp1_2 = filter_and_reindex(experience_1_2_valid, root_dir).rename(
    columns={'img_path': 'image_path'}
)
filtered_test_data_exp1_2 = filter_and_reindex(experience_1_2_test, root_dir).rename(
    columns={'img_path': 'image_path'}
)

# Create dataset instances for combined Experience 1_2.
train_dataset_exp1_2 = NaiveCompatibleBalancedDataset(
    data_frame=filtered_train_data_exp1_2,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
val_dataset_exp1_2 = NaiveCompatibleBalancedDataset(
    data_frame=filtered_valid_data_exp1_2,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
test_dataset_exp1_2 = NaiveCompatibleBalancedDataset(
    data_frame=filtered_test_data_exp1_2,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)

# ------------------
# Process Experience 3
# ------------------
# Use the already split DataFrames for Experience 3.
filtered_train_data_exp3 = filter_and_reindex(experience_3_train, root_dir).rename(
    columns={'img_path': 'image_path'}
)
filtered_valid_data_exp3 = filter_and_reindex(experience_3_valid, root_dir).rename(
    columns={'img_path': 'image_path'}
)
filtered_test_data_exp3 = filter_and_reindex(experience_3_test, root_dir).rename(
    columns={'img_path': 'image_path'}
)

train_dataset_exp3 = NaiveCompatibleBalancedDataset(
    data_frame=filtered_train_data_exp3,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
val_dataset_exp3 = NaiveCompatibleBalancedDataset(
    data_frame=filtered_valid_data_exp3,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
test_dataset_exp3 = NaiveCompatibleBalancedDataset(
    data_frame=filtered_test_data_exp3,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)


# ## Creating Dataloaders for more efficient data processing

# In[ ]:


from torch.utils.data.dataloader import DataLoader

# ---------------------------
# Experience 1_2: Combined Data
# ---------------------------
train_sampler_exp1_2 = BalancedBatchSampler(
    data_frame=filtered_train_data_exp1_2, 
    batch_size=10, 
    samples_per_class=5
)
val_sampler_exp1_2 = BalancedBatchSampler(
    data_frame=filtered_valid_data_exp1_2, 
    batch_size=10, 
    samples_per_class=5
)
test_sampler_exp1_2 = BalancedBatchSampler(
    data_frame=filtered_test_data_exp1_2, 
    batch_size=10, 
    samples_per_class=5
)

train_loader_exp1_2 = DataLoader(train_dataset_exp1_2, batch_sampler=train_sampler_exp1_2, shuffle=False)
val_loader_exp1_2 = DataLoader(val_dataset_exp1_2, batch_sampler=val_sampler_exp1_2, shuffle=False)
test_loader_exp1_2 = DataLoader(test_dataset_exp1_2, batch_sampler=test_sampler_exp1_2, shuffle=False)

# ---------------------------
# Experience 3: Original Data
# ---------------------------
train_sampler_exp3 = BalancedBatchSampler(
    data_frame=filtered_train_data_exp3, 
    batch_size=10, 
    samples_per_class=5
)
val_sampler_exp3 = BalancedBatchSampler(
    data_frame=filtered_valid_data_exp3, 
    batch_size=10, 
    samples_per_class=5
)
test_sampler_exp3 = BalancedBatchSampler(
    data_frame=filtered_test_data_exp3, 
    batch_size=10, 
    samples_per_class=5
)

train_loader_exp3 = DataLoader(train_dataset_exp3, batch_sampler=train_sampler_exp3, shuffle=False)
val_loader_exp3 = DataLoader(val_dataset_exp3, batch_sampler=val_sampler_exp3, shuffle=False)
test_loader_exp3 = DataLoader(test_dataset_exp3, batch_sampler=test_sampler_exp3, shuffle=False)

print("DataLoaders for all experiences created successfully!")


# ## Checking class distribution in each dataset

# In[ ]:


import torch
from collections import Counter

def count_classes(dataset):
    # Convert the targets attribute into a list of values.
    values = [x for x in dataset.targets]
    # Convert the list of values to a tensor.
    t = torch.tensor(values)
    # Convert the tensor to a NumPy array and count the classes.
    return Counter(t.numpy())

print("Class distribution in Train Dataset (Experience 1_2):", count_classes(train_dataset_exp1_2))
print("Class distribution in Train Dataset (Experience 3):", count_classes(train_dataset_exp3))

print("Class distribution in Validation Dataset (Experience 1_2):", count_classes(val_dataset_exp1_2))
print("Class distribution in Validation Dataset (Experience 3):", count_classes(val_dataset_exp3))

print("Class distribution in Test Dataset (Experience 1_2):", count_classes(test_dataset_exp1_2))
print("Class distribution in Test Dataset (Experience 3):", count_classes(test_dataset_exp3))


# ## Checking unique classes in each experience

# In[ ]:


from avalanche.benchmarks.utils import DataAttribute
from avalanche.benchmarks import benchmark_from_datasets

# Create the benchmark from your datasets using the combined experience_1_2 and experience_3.
dataset_streams = {
    "train": [train_dataset_exp1_2, train_dataset_exp3],
    "test": [test_dataset_exp1_2, test_dataset_exp3]
}

benchmark = benchmark_from_datasets(**dataset_streams)

for experience in benchmark.train_stream:
    print(f"Start of experience: {experience.current_experience}")
    
    # Try to get the targets via the dynamic property.
    try:
        targets_data = experience.dataset.targets.data
    except AttributeError:
        # Fallback: access the internal _data_attributes dictionary.
        targets_data = experience.dataset._data_attributes["targets"].data

    # If targets_data doesn't have 'tolist', assume it's already iterable.
    if hasattr(targets_data, "tolist"):
        unique_classes = set(targets_data.tolist())
    else:
        unique_classes = set(targets_data)
        
    print(f"Classes in this experience: {unique_classes}")


# ## Implementing Naive strategy using Avalanche - the end-to-end continual learning library

# In[ ]:


import os
import csv
import itertools
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    forgetting_metrics,
    StreamConfusionMatrix,
    disk_usage_metrics
)
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.benchmarks.utils import DataAttribute
from models.cnn_models import SimpleCNN, update_classifier  # Import both your CNN and update_classifier

# -------------------------------
# Create main folder for experiment outputs
# -------------------------------
MAIN_OUT_FOLDER = "naive_experiment"
os.makedirs(MAIN_OUT_FOLDER, exist_ok=True)

# -------------------------------
# Helper function: log metrics to CSV
# -------------------------------
def log_metrics(csv_file, experience_id, epoch, train_loss, train_acc, val_loss, val_acc):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experience_id, epoch, train_loss, train_acc, val_loss, val_acc])

# -------------------------------
# Helper function: plot metrics and save to folder "loss_plots"
# -------------------------------
def plot_metrics(epochs, train_vals, val_vals, ylabel, title, filename):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_vals, 'b-', label=f'Train {ylabel}')
    plt.plot(epochs, val_vals, 'r-', label=f'Validation {ylabel}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

# -------------------------------
# Helper function: save a confusion matrix as an image.
# -------------------------------
def save_confusion_matrix(cm, title, filename):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(cm.shape[0])
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2.0 else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

# -------------------------------
# Setup loggers and device
# -------------------------------
tb_logger = TensorboardLogger()
text_logger = TextLogger(open('log.txt', 'a'))
interactive_logger = InteractiveLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Setup benchmark and validation datasets
# -------------------------------
# Note: Here, we assume you have already created the following datasets:
#   - train_dataset_exp1_2, val_dataset_exp1_2, test_dataset_exp1_2  (for combined experiences 1 & 2)
#   - train_dataset_exp3,    val_dataset_exp3,    test_dataset_exp3       (for experience 3)

dataset_streams = {
    "train": [train_dataset_exp1_2, train_dataset_exp3],
    "test": [test_dataset_exp1_2, test_dataset_exp3]
}
benchmark = benchmark_from_datasets(**dataset_streams)
# Also store the validation datasets for later use.
validation_datasets = [val_dataset_exp1_2, val_dataset_exp3]

# -------------------------------
# Set learning rate and prepare results summary
# -------------------------------
lr = 0.001
results_summary = []

# Create a folder for this hyperparameter configuration.
config_folder = os.path.join(MAIN_OUT_FOLDER, f"lr{lr}")
os.makedirs(config_folder, exist_ok=True)

# Prepare a CSV file for summary metrics for this configuration.
csv_file_path = os.path.join(config_folder, f"summary_lr{lr}.csv")
with open(csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Experience", "Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc"])

# We'll maintain the model between experiences.
model = None

# For each experience in the benchmark.
for exp_idx, experience in enumerate(benchmark.train_stream):
    print(f"\n=== Start of Experience {experience.current_experience} ===")
    
    # Select the correct DataLoaders based on the experience.
    # Experience 0: Combined experiences 1 & 2 (2 classes)
    # Experience 1: Experience 3 (introduces the new class)
    if experience.current_experience == 0:
        current_train_loader = train_loader_exp1_2
        current_val_loader   = val_loader_exp1_2
        current_test_loader  = test_loader_exp1_2
        # Instantiate model with 2 output classes.
        model = SimpleCNN(num_classes=2).to(device)
    elif experience.current_experience == 1:
        current_train_loader = train_loader_exp3
        current_val_loader   = val_loader_exp3
        current_test_loader  = test_loader_exp3
        # Update classifier to support 3 classes while preserving learned weights.
        model = update_classifier(model, new_num_classes=3)
    else:
        raise ValueError("Unexpected experience id")
    
    # Create a folder for this experience.
    exp_folder = os.path.join(config_folder, f"experience_{experience.current_experience}")
    os.makedirs(exp_folder, exist_ok=True)
    
    # Create a validation benchmark using the validation dataset.
    val_benchmark = benchmark_from_datasets(
        train=[current_val_loader.dataset],
        test=[current_val_loader.dataset]
    )
    
    # Set up criterion, optimizer, and learning rate scheduler.
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    lr_plugin = LRSchedulerPlugin(scheduler)
    
    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        # Even though experience 0 has 2 classes, we set num_classes=3 overall.
        StreamConfusionMatrix(num_classes=3, save_image=False),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    # Instantiate the Naive strategy.
    # Note: We set train_epochs=1 so we can loop for each epoch.
    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=15,
        train_epochs=1,
        eval_mb_size=15,
        evaluator=evaluator,
        eval_every=-1,  # We'll perform our own per-epoch evaluation.
        device=device,
        plugins=[lr_plugin]
    )
    
    # Lists to store per-epoch metrics.
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    num_epochs = 10  # Adjust this as needed.
    
    # Training and validation loop.
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} for Experience {experience.current_experience} ...")
        train_res = cl_strategy.train(experience, train_loader=current_train_loader)
        epoch_train_loss = train_res.get("Loss_Epoch/train_phase/train_stream", None)
        epoch_train_acc  = train_res.get("Top1_Acc_Epoch/train_phase/train_stream", None)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        
        # Evaluate on the validation dataset.
        val_res = cl_strategy.eval(val_benchmark.test_stream)
        epoch_val_loss = val_res.get("Loss_Stream/eval_phase/test_stream", None)
        epoch_val_acc  = val_res.get("Top1_Acc_Stream/eval_phase/test_stream", None)
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)
        
        print(f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
        
        # Step the scheduler.
        scheduler.step()
        
        # Log the epoch's metrics.
        log_metrics(csv_file_path, experience.current_experience, epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)
    
    # Plot losses and accuracies.
    epochs_range = list(range(1, num_epochs + 1))
    loss_title = f"Exp {experience.current_experience}: lr={lr} (Loss)"
    loss_plot_path = os.path.join(exp_folder, f"loss_plot_exp{experience.current_experience}.png")
    plot_metrics(epochs_range, train_loss_history, val_loss_history, "Loss", loss_title, loss_plot_path)
    
    acc_title = f"Exp {experience.current_experience}: lr={lr} (Accuracy)"
    acc_plot_path = os.path.join(exp_folder, f"acc_plot_exp{experience.current_experience}.png")
    plot_metrics(epochs_range, train_acc_history, val_acc_history, "Accuracy", acc_title, acc_plot_path)
    
    # --- End-of-Experience Testing ---
    print("Testing on each test dataset for this experience...")
    # Evaluate on each test dataset in the benchmark.
    for test_idx, test_dataset in enumerate(dataset_streams["test"]):
        test_benchmark = benchmark_from_datasets(
            train=[test_dataset],
            test=[test_dataset]
        )
        test_results = cl_strategy.eval(test_benchmark.test_stream)
        test_cm = test_results.get("ConfusionMatrix_Stream/eval_phase/test_stream", None)
        if test_cm is not None and test_cm != "No confusion matrix available":
            try:
                cm_array = np.array(test_cm)
                filename = os.path.join(exp_folder, f"test_confusion_matrix_dataset_{test_idx}.png")
                title = f"Experience {experience.current_experience} Test Confusion Matrix for Test Dataset {test_idx}"
                save_confusion_matrix(cm_array, title, filename)
                print(f"Saved confusion matrix for experience {experience.current_experience}, test dataset {test_idx} to {filename}")
            except Exception as e:
                print(f"Could not save confusion matrix for experience {experience.current_experience}, test dataset {test_idx}: {e}")
        else:
            print(f"No confusion matrix available for experience {experience.current_experience}, test dataset {test_idx}")
    
    # Optionally, evaluate on the entire test stream.
    print("Evaluating on the entire test stream...")
    test_res = cl_strategy.eval(benchmark.test_stream)
    print("Test results:", test_res)
    
    results_summary.append({
        "lr": lr,
        "final_train_loss": train_loss_history[-1],
        "final_val_loss": val_loss_history[-1],
        "test_results": test_res
    })

print("\n=== Hyperparameter Search Summary ===")
for res in results_summary:
    print(res)


# In[ ]:




