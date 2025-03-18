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
pattern = 'print24|print131|print0|print46|print82|print111|print132'
data_filtered = data[data.iloc[:, 0].str.contains(pattern, na=False)]

# Update the first column to include both the print folder and the image filename.
# The regex now captures the folder name (print24, print131, or print0) and the image filename.
data_filtered.iloc[:, 0] = data_filtered.iloc[:, 0].str.replace(
    r'.*?/(print24|print131|print0|print46|print82|print111|print132)/(image-\d+\.jpg)', 
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
    desired_size = 15000
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
test_prop = 0.1  # Not explicitly used; computed as remainder

# For each experience, split the dataset using all available images.
for exp_id in [1, 2, 3]:
    if exp_id not in experience_datasets:
        continue

    # Work only on the necessary columns
    dataset = experience_datasets[exp_id][['img_path', 'hotend_class']]
    
    # Shuffle the dataset indices
    indices = dataset.index.tolist()
    random.shuffle(indices)
    
    total_images = len(indices)
    train_count = int(train_prop * total_images)
    valid_count = int(valid_prop * total_images)
    # The test set gets the remaining images
    test_count = total_images - train_count - valid_count
    
    # Split the indices according to the computed counts
    train_indices = indices[:train_count]
    valid_indices = indices[train_count:train_count + valid_count]
    test_indices  = indices[train_count + valid_count:]
    
    # Create the train, validation, and test DataFrames
    globals()[f'train_{exp_id}'] = dataset.loc[train_indices].reset_index(drop=True)
    globals()[f'valid_{exp_id}'] = dataset.loc[valid_indices].reset_index(drop=True)
    globals()[f'test_{exp_id}']  = dataset.loc[test_indices].reset_index(drop=True)
    
    print(f"\n--- Experience {exp_id} Splits ---")
    print(f"Total images in Experience {exp_id}: {total_images}")
    print(f"Train set size: {len(globals()[f'train_{exp_id}'])}")
    print(f"Validation set size: {len(globals()[f'valid_{exp_id}'])}")
    print(f"Test set size: {len(globals()[f'test_{exp_id}'])}")
    print("Class distribution (in entire experience dataset):", dataset['hotend_class'].value_counts().to_dict())


# ## BalancedBatchSampler Class

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


# ## Build global indices function

# In[ ]:


def build_global_class_indices(concat_dataset):
    """
    Builds a dictionary mapping each class (from 'hotend_class') to a list of global indices 
    for the given ConcatDataset.
    """
    global_class_indices = {}
    offset = 0
    # Iterate over each sub-dataset in the ConcatDataset.
    for ds in concat_dataset.datasets:
        # Assume each sub-dataset has a 'data' attribute (a DataFrame).
        df = ds.data.reset_index(drop=True) if hasattr(ds, 'data') else ds.reset_index(drop=True)
        for cls in df['hotend_class'].unique():
            if cls not in global_class_indices:
                global_class_indices[cls] = []
        # Add global indices (local index + offset)
        for local_idx, row in df.iterrows():
            cls = row['hotend_class']
            global_class_indices[cls].append(offset + local_idx)
        offset += len(df)
    return global_class_indices

class GlobalBalancedBatchSampler(Sampler):
    """
    A sampler that yields balanced batches using global indices computed for a ConcatDataset.
    It guarantees that each batch contains exactly `samples_per_class` samples from every class.
    If a class does not have enough remaining indices in the current epoch, it will sample with replacement.
    """
    def __init__(self, global_class_indices, batch_size=15, samples_per_class=5):
        """
        Args:
            global_class_indices (dict): Mapping from class label to list of global indices.
            batch_size (int): Total batch size (must be divisible by the number of classes).
            samples_per_class (int): Number of samples to draw per class in each batch.
        """
        self.global_class_indices = global_class_indices
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_classes = len(global_class_indices)
        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")

    def __iter__(self):
        # Make a local copy of the global indices for each class.
        indices_used = {cls: self.global_class_indices[cls].copy() for cls in self.global_class_indices}
        # Shuffle each list.
        for cls in indices_used:
            random.shuffle(indices_used[cls])
        
        # Compute the number of batches based on the minimum available indices.
        # (This value may be less than what you want, but we will use replacement for classes that run low.)
        num_batches = min(len(indices) for indices in indices_used.values()) // self.samples_per_class
        
        batches = []
        # Always iterate over sorted keys to ensure a consistent order.
        class_keys = sorted(self.global_class_indices.keys())
        for _ in range(num_batches):
            batch = []
            for cls in class_keys:
                # If there aren’t enough remaining indices for this class, sample with replacement.
                if len(indices_used[cls]) < self.samples_per_class:
                    sampled = random.choices(self.global_class_indices[cls], k=self.samples_per_class)
                else:
                    sampled = indices_used[cls][:self.samples_per_class]
                    indices_used[cls] = indices_used[cls][self.samples_per_class:]
                batch.extend(sampled)
            random.shuffle(batch)
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        # Return the number of batches computed from the minimum count (without replacement).
        return min(len(indices) for indices in self.global_class_indices.values()) // self.samples_per_class


# ## Creating experience datasets

# In[ ]:


from torch.utils.data import ConcatDataset

def wrap_dataset(ds, root_dir):
    if hasattr(ds, 'data'):
        df = ds.data.reset_index(drop=True)
    else:
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


from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd

def create_balanced_loader(dataset, root_dir, batch_size=15, samples_per_class=None):
    """
    Given an experience dataset (plain DataFrame, BalancedDataset, or ConcatDataset),
    create a DataLoader using a balanced batch sampler built from the underlying data.
    
    For single-class datasets, the entire batch (of size batch_size) is used.
    For multi-class datasets, each batch will contain samples_per_class samples from each class.
    If samples_per_class is not provided for multi-class datasets, a default of 5 is used.
    
    If the provided batch_size is not divisible by the number of classes in a multi-class dataset,
    it will be adjusted to be samples_per_class * num_classes.
    """
    # If the dataset is a plain DataFrame, wrap it.
    if isinstance(dataset, pd.DataFrame):
        dataset = BalancedDataset(dataset, root_dir, debug=False)
    
    # Check dataset type and adjust parameters accordingly.
    if isinstance(dataset, ConcatDataset):
        global_class_indices = build_global_class_indices(dataset)
        num_classes = len(global_class_indices)
        # Determine samples per class: for single-class use full batch, otherwise use provided or default 5.
        spc = batch_size if num_classes == 1 else (samples_per_class if samples_per_class is not None else 5)
        # For multi-class datasets, ensure batch_size is exactly spc * num_classes.
        if num_classes > 1 and batch_size % num_classes != 0:
            adjusted_batch_size = spc * num_classes
            print(f"Adjusted batch_size from {batch_size} to {adjusted_batch_size} for balanced sampling across {num_classes} classes.")
            batch_size = adjusted_batch_size
        sampler = GlobalBalancedBatchSampler(global_class_indices, batch_size=batch_size, samples_per_class=spc)
    elif hasattr(dataset, 'data'):
        data_for_sampler = dataset.data.reset_index(drop=True)
        unique_classes = data_for_sampler['hotend_class'].unique()
        num_classes = len(unique_classes)
        spc = batch_size if num_classes == 1 else (samples_per_class if samples_per_class is not None else 5)
        if num_classes > 1 and batch_size % num_classes != 0:
            adjusted_batch_size = spc * num_classes
            print(f"Adjusted batch_size from {batch_size} to {adjusted_batch_size} for balanced sampling across {num_classes} classes.")
            batch_size = adjusted_batch_size
        sampler = BalancedBatchSampler(data_frame=data_for_sampler, batch_size=batch_size, samples_per_class=spc)
    else:
        raise ValueError("Dataset type not recognized for sampler creation.")
    
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


# ## Checking class distribution in each dataset

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


import re
import pandas as pd
from torch.utils.data import ConcatDataset

def extract_folder(file_path):
    """
    Extracts the folder name from the file path that matches the pattern 'print' followed by digits.
    For example, from '/some/path/print24/image.jpg' it returns 'print24'.
    """
    match = re.search(r'(print\d+)', file_path)
    if match:
        return match.group(1)
    return None

def get_unique_folders(dataset):
    """
    Returns a set of unique folder names extracted from the image paths in the dataset.
    It supports plain DataFrames, ConcatDataset, and custom datasets with a 'data' attribute.
    Assumes the image path is stored in a column named 'image_path'.
    """
    folders = set()
    
    def process_df(df):
        col_name = 'img_path'  # Update to the correct column name if necessary
        if col_name not in df.columns:
            raise KeyError(f"Expected column '{col_name}' not found in DataFrame columns: {list(df.columns)}")
        for path in df[col_name]:
            folder = extract_folder(path)
            if folder:
                folders.add(folder)
    
    if isinstance(dataset, pd.DataFrame):
        process_df(dataset)
    elif isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, 'data'):
                process_df(ds.data.reset_index(drop=True))
            else:
                process_df(ds.reset_index(drop=True))
    elif hasattr(dataset, 'data'):
        process_df(dataset.data.reset_index(drop=True))
    else:
        raise ValueError("Dataset type not recognized.")
    
    return folders

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

# Now, print the sizes, class distributions, and unique folders for your experience datasets.

# For Experience 1:
print("Size of Experience 1 train dataset:", get_dataset_size(exp1_train))
print("Class distribution in Experience 1 train dataset:", count_classes(exp1_train))
print("Unique folders in Experience 1 train dataset:", get_unique_folders(exp1_train))
print("Size of Experience 1 valid dataset:", get_dataset_size(exp1_valid))
print("Class distribution in Experience 1 valid dataset:", count_classes(exp1_valid))
print("Unique folders in Experience 1 valid dataset:", get_unique_folders(exp1_valid))
print("Size of Experience 1 test dataset:", get_dataset_size(exp1_test))
print("Class distribution in Experience 1 test dataset:", count_classes(exp1_test))
print("Unique folders in Experience 1 test dataset:", get_unique_folders(exp1_test))

# For Experience 1_2:
print("Size of Experience 1_2 train dataset:", get_dataset_size(exp1_2_train))
print("Class distribution in Experience 1_2 train dataset:", count_classes(exp1_2_train))
print("Unique folders in Experience 1_2 train dataset:", get_unique_folders(exp1_2_train))
print("Size of Experience 1_2 valid dataset:", get_dataset_size(exp1_2_valid))
print("Class distribution in Experience 1_2 valid dataset:", count_classes(exp1_2_valid))
print("Unique folders in Experience 1_2 valid dataset:", get_unique_folders(exp1_2_valid))
print("Size of Experience 1_2 test dataset:", get_dataset_size(exp1_2_test))
print("Class distribution in Experience 1_2 test dataset:", count_classes(exp1_2_test))
print("Unique folders in Experience 1_2 test dataset:", get_unique_folders(exp1_2_test))

# For Experience 1_2_3:
print("Size of Experience 1_2_3 train dataset:", get_dataset_size(exp1_2_3_train))
print("Class distribution in Experience 1_2_3 train dataset:", count_classes(exp1_2_3_train))
print("Unique folders in Experience 1_2_3 train dataset:", get_unique_folders(exp1_2_3_train))
print("Size of Experience 1_2_3 valid dataset:", get_dataset_size(exp1_2_3_valid))
print("Class distribution in Experience 1_2_3 valid dataset:", count_classes(exp1_2_3_valid))
print("Unique folders in Experience 1_2_3 valid dataset:", get_unique_folders(exp1_2_3_valid))
print("Size of Experience 1_2_3 test dataset:", get_dataset_size(exp1_2_3_test))
print("Class distribution in Experience 1_2_3 test dataset:", count_classes(exp1_2_3_test))
print("Unique folders in Experience 1_2_3 test dataset:", get_unique_folders(exp1_2_3_test))


# ## Checking image and label alignment

# In[ ]:


import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import ConcatDataset
import pandas as pd  # make sure to import pandas

def display_random_images(dataset, dataset_name, num_images=5, root_folder=r"C:\Users\Sandhra George\avalanche\caxton_dataset"):
    """
    Display num_images random images from the given dataset along with their hotend class labels.
    
    If the underlying DataFrame has 2 columns:
      - Column 0: relative image path (e.g., "print46/image-8719.jpg")
      - Column 1: hotend class label
    If the DataFrame has 3 (or more) columns, it ignores the first column and uses:
      - Column 1: relative image path
      - Column 2: hotend class label
    
    The full image path is constructed as:
         root_folder/<relative image path>
    
    If the dataset is a ConcatDataset, the function iterates over its sub-datasets.
    """
    
    # Check if dataset is a pandas DataFrame or has a .data attribute.
    if isinstance(dataset, pd.DataFrame):
        data = dataset
    elif hasattr(dataset, 'data'):
        data = dataset.data
    elif isinstance(dataset, ConcatDataset):
        print(f"Dataset '{dataset_name}' is a ConcatDataset; iterating through sub-datasets...")
        for i, subdataset in enumerate(dataset.datasets):
            display_random_images(subdataset, f"{dataset_name} - Subdataset {i}", num_images, root_folder)
        return
    else:
        print("Unsupported dataset type:", type(dataset))
        return

    cols = data.shape[1]
    if cols == 2:
        # Structure: [image_path, hotend_label]
        get_img_path = lambda row: row[0]
        get_label   = lambda row: row[1]
    elif cols >= 3:
        # Structure: [extra, image_path, hotend_label, ...] – ignore the first column
        get_img_path = lambda row: row[1]
        get_label   = lambda row: row[2]
    else:
        print(f"Dataset {dataset_name} does not have enough columns (got {cols}).")
        return

    print(f"Displaying {num_images} random images from: {dataset_name}")
    sample_indices = random.sample(range(len(data)), num_images)
    for idx in sample_indices:
        row = data.iloc[idx]
        # Clean the strings: remove extra spaces and ensure correct separator
        img_rel_path = str(get_img_path(row)).strip().replace("/", os.path.sep)
        label = str(get_label(row)).strip()
        full_path = os.path.join(root_folder, img_rel_path)
        
        # Debug print: print the full path with repr to reveal any hidden characters
        print(f"Attempting to open: {repr(full_path)}")
        
        if not os.path.exists(full_path):
            print(f"File does not exist: {repr(full_path)}")
            continue
        
        try:
            img = Image.open(full_path)
        except Exception as e:
            print(f"Error opening {full_path}: {e}")
            continue

        plt.figure()
        plt.imshow(img)
        plt.title(f"Hotend Class: {label}")
        plt.axis("off")
        plt.show()

# Example usage:
display_random_images(exp1_train, "Experience 1 Train Dataset")
display_random_images(exp1_valid, "Experience 1 Valid Dataset")
display_random_images(exp1_test, "Experience 1 Test Dataset")

display_random_images(exp1_2_train, "Experience 1_2 Train Dataset")
display_random_images(exp1_2_valid, "Experience 1_2 Valid Dataset")
display_random_images(exp1_2_test, "Experience 1_2 Test Dataset")

display_random_images(exp1_2_3_train, "Experience 1_2_3 Train Dataset")
display_random_images(exp1_2_3_valid, "Experience 1_2_3 Valid Dataset")
display_random_images(exp1_2_3_test, "Experience 1_2_3 Test Dataset")


# ## Benchmark experiment 

# In[ ]:


import os
import csv
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import ConfusionMatrix
from models.cnn_models import SimpleCNN
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
#   "experience_1" --> exp1_train_loader, exp1_valid_loader, exp1_test_loader
#   "experience_1_2" --> exp1_2_train_loader, exp1_2_valid_loader, exp1_2_test_loader
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
    
    # ----------------- Checkpoint Loading -----------------
    if os.path.exists(best_model_path):
        print(f"Loading checkpoint from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.4f}")
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        print("No checkpoint found, starting fresh.")
    
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
    
    # ----------------- Training and Validation Loop -----------------
    for epoch in range(start_epoch, num_epochs):
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
            #print(f"Outputs (Raw): {outputs}")  # Log raw outputs
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
            #for i in range(len(labels)):
                #print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
        
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
                #print(f"Outputs (Raw): {outputs}")  # Log raw outputs
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)
                
                val_cm.update(predicted, labels)
                for label in labels:
                    val_class_counts[label.item()] += 1
                
                #for i in range(len(labels)):
                    #print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
        
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
                #for i in range(len(labels)):
                    #print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
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


# In[ ]:




