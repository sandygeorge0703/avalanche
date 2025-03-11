#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[58]:


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

# In[59]:


# Define file paths as constants
CSV_FILE_PATH = r'C:\Users\Sandhra George\avalanche\data\dataset.csv'
ROOT_DIR_PATH = r'C:\Users\Sandhra George\avalanche\caxton_dataset'  # Common parent directory

# Load data into a DataFrame for easier processing
data = pd.read_csv(CSV_FILE_PATH)

# Filter the dataset to include images containing "print24", "print131", or "print0"
pattern = 'print24|print131|print0|print46|print82|print109|print111|print132|print171'
data_filtered = data[data.iloc[:, 0].str.contains(pattern, na=False)]

# Update the first column to include both the print folder and the image filename.
# The regex now captures the folder name (print24, print131, or print0) and the image filename.
data_filtered.iloc[:, 0] = data_filtered.iloc[:, 0].str.replace(
    r'.*?/(print24|print131|print0|print46|print82|print109|print111|print132|print171)/(image-\d+\.jpg)', 
    r'\1/\2', 
    regex=True
)

# Display the updated DataFrame
print("First rows of filtered DataFrame:")
print(data_filtered.head())

print("\nLast rows of filtered DataFrame:")
print(data_filtered.tail())


# ## Analysing the target hotend temperature column

# In[60]:


unique_temperatures = sorted(data_filtered['target_hotend'].unique())

if len(unique_temperatures) >= 69:
    temperature_min = unique_temperatures[0]
    temperature_max = unique_temperatures[-1]
    remaining_temperatures = [temp for temp in unique_temperatures if temp not in [temperature_min, temperature_max]]
    random_temperatures = random.sample(remaining_temperatures, 50)
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

# Create a dictionary to store balanced datasets (non-cumulative) for each experience
experience_datasets = {}

for exp_id, experience_temps in enumerate([experience_1, experience_2, experience_3], start=1):
    if not experience_temps:
        print(f"Skipping Experience {exp_id} due to insufficient temperatures.")
        continue
    print(f"\nProcessing Experience {exp_id} with temperatures: {experience_temps}...")
    
    # Filter data for the current experience's temperatures
    exp_data = data_filtered[data_filtered['target_hotend'].isin(experience_temps)]
    if exp_data.empty:
        print(f"No data found for Experience {exp_id}. Skipping...")
        continue
    
    # Create dictionary for each class (assumed classes: 0, 1, 2)
    class_datasets = {}
    for class_id in [0, 1, 2]:
        class_data = exp_data[exp_data['hotend_class'] == class_id]
        if class_data.empty:
            print(f"Warning: Class {class_id} in Experience {exp_id} has no data!")
        else:
            class_datasets[class_id] = class_data
    
    if len(class_datasets) != 3:
        print(f"Skipping Experience {exp_id} because one or more classes are missing data!")
        continue
    
    # Balance by sampling the minimum available images per class
    min_class_size = min(len(class_datasets[c]) for c in class_datasets)
    print(f"Smallest class size in Experience {exp_id}: {min_class_size}")
    
    balanced_data = [class_datasets[c].sample(n=min_class_size, random_state=42) for c in class_datasets]
    balanced_dataset = pd.concat(balanced_data).reset_index(drop=True)
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    experience_datasets[exp_id] = balanced_dataset
    print(f"Balanced dataset size for Experience {exp_id}: {len(balanced_dataset)}")
    for class_id in [0,1,2]:
        count = len(balanced_dataset[balanced_dataset['hotend_class'] == class_id])
        print(f"Class {class_id} count: {count}")


# In[61]:


# Determine the overall minimum number of images per class across all experiences
min_images_per_class_overall = min(
    [min(experience_datasets[exp]['hotend_class'].value_counts()) for exp in experience_datasets]
)
print("Overall minimum images per class across experiences:", min_images_per_class_overall)

# Define split proportions
train_prop = 0.7
valid_prop = 0.15
test_prop = 0.15

samples_per_class_train = int(train_prop * min_images_per_class_overall)
samples_per_class_valid = int(valid_prop * min_images_per_class_overall)
# The test set gets the remaining images
samples_per_class_test  = min_images_per_class_overall - samples_per_class_train - samples_per_class_valid

print("Samples per class - Training:", samples_per_class_train)
print("Samples per class - Validation:", samples_per_class_valid)
print("Samples per class - Test:", samples_per_class_test)

# For each experience, re-sample the balanced dataset accordingly.
for exp_id in [1, 2, 3]:
    if exp_id not in experience_datasets:
        continue
    # Work only on the necessary columns
    balanced_dataset_filtered = experience_datasets[exp_id][['img_path', 'hotend_class']]
    
    train_indices, valid_indices, test_indices = [], [], []
    for class_label in [0, 1, 2]:
        # Get indices for current class
        class_indices = balanced_dataset_filtered[balanced_dataset_filtered['hotend_class'] == class_label].index.tolist()
        random.shuffle(class_indices)
        train_indices.extend(class_indices[:samples_per_class_train])
        valid_indices.extend(class_indices[samples_per_class_train:samples_per_class_train + samples_per_class_valid])
        test_indices.extend(class_indices[samples_per_class_train + samples_per_class_valid:
                                           samples_per_class_train + samples_per_class_valid + samples_per_class_test])
    
    # Sort indices (optional, for consistency)
    train_indices = sorted(train_indices)
    valid_indices = sorted(valid_indices)
    test_indices = sorted(test_indices)
    
    globals()[f'train_{exp_id}'] = balanced_dataset_filtered.loc[train_indices].reset_index(drop=True)
    globals()[f'valid_{exp_id}'] = balanced_dataset_filtered.loc[valid_indices].reset_index(drop=True)
    globals()[f'test_{exp_id}']  = balanced_dataset_filtered.loc[test_indices].reset_index(drop=True)
    
    print(f"\n--- Experience {exp_id} Splits ---")
    print(f"Train set size: {len(globals()[f'train_{exp_id}'])} (Expected: {samples_per_class_train*3})")
    print(f"Validation set size: {len(globals()[f'valid_{exp_id}'])} (Expected: {samples_per_class_valid*3})")
    print(f"Test set size: {len(globals()[f'test_{exp_id}'])} (Expected: {samples_per_class_test*3})")
    for split in ['train', 'valid', 'test']:
        df = globals()[f'{split}_{exp_id}']
        counts = df['hotend_class'].value_counts().to_dict()
        print(f"{split.capitalize()} class distribution: {counts}")


# ## Balanced Batch Sampler class

# In[62]:


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

# In[63]:


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

# In[64]:


def filter_and_reindex(data_frame, root_dir):
    """
    Filters the DataFrame to include only rows with valid image paths
    and then reindexes the DataFrame so that indices are contiguous.
    """
    valid_indices = []
    allowed_folders = {"print24", "print131", "print0", "print46","print82","print109","print111","print132","print172"}
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


# ## Function to print random dataset batches

# In[65]:


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


# In[66]:


# Define the root directory
ROOT_DIR_PATH = r'C:\Users\Sandhra George\avalanche\caxton_dataset'
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

# In[67]:


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


# ## Creating an EWC Class which inherits from AvalancheDataset and contains all the expected functions

# In[68]:


import os
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute
from avalanche.benchmarks.utils.transforms import TupleTransform

class EWCCompatibleBalancedDataset(AvalancheDataset):
    def __init__(self, data_frame, root_dir=None, transform=None, task_label=0, indices=None):
        """
        Custom dataset compatible with EWC that inherits from AvalancheDataset.
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

# In[69]:


from torchvision import transforms

# Define the transformation (e.g., normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Experience 1
filtered_train_data_exp1 = filter_and_reindex(train_1, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)
filtered_valid_data_exp1 = filter_and_reindex(valid_1, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)
filtered_test_data_exp1 = filter_and_reindex(test_1, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)

train_dataset_exp1 = EWCCompatibleBalancedDataset(
    data_frame=filtered_train_data_exp1,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
val_dataset_exp1 = EWCCompatibleBalancedDataset(
    data_frame=filtered_valid_data_exp1,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
test_dataset_exp1 = EWCCompatibleBalancedDataset(
    data_frame=filtered_test_data_exp1,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)

# Experience 2
filtered_train_data_exp2 = filter_and_reindex(train_2, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)
filtered_valid_data_exp2 = filter_and_reindex(valid_2, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)
filtered_test_data_exp2 = filter_and_reindex(test_2, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)

train_dataset_exp2 = EWCCompatibleBalancedDataset(
    data_frame=filtered_train_data_exp2,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
val_dataset_exp2 = EWCCompatibleBalancedDataset(
    data_frame=filtered_valid_data_exp2,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
test_dataset_exp2 = EWCCompatibleBalancedDataset(
    data_frame=filtered_test_data_exp2,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)

# Experience 3
filtered_train_data_exp3 = filter_and_reindex(train_3, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)
filtered_valid_data_exp3 = filter_and_reindex(valid_3, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)
filtered_test_data_exp3 = filter_and_reindex(test_3, root_dir).rename(
    columns={'img_path': 'image_path', 'class': 'hotend_class'}
)

train_dataset_exp3 = EWCCompatibleBalancedDataset(
    data_frame=filtered_train_data_exp3,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
val_dataset_exp3 = EWCCompatibleBalancedDataset(
    data_frame=filtered_valid_data_exp3,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)
test_dataset_exp3 = EWCCompatibleBalancedDataset(
    data_frame=filtered_test_data_exp3,
    root_dir=root_dir,
    transform=transform,
    task_label=0
)


# ## Creating Dataloaders for more efficient data processing

# In[70]:


from torch.utils.data.dataloader import DataLoader

# Experience 1: using the filtered DataFrames
train_sampler_exp1 = BalancedBatchSampler(
    data_frame=filtered_train_data_exp1, 
    batch_size=15, 
    samples_per_class=5
)
val_sampler_exp1 = BalancedBatchSampler(
    data_frame=filtered_valid_data_exp1, 
    batch_size=15, 
    samples_per_class=5
)
test_sampler_exp1 = BalancedBatchSampler(
    data_frame=filtered_test_data_exp1, 
    batch_size=15, 
    samples_per_class=5
)

train_loader_exp1 = DataLoader(train_dataset_exp1, batch_sampler=train_sampler_exp1, shuffle=False)
val_loader_exp1 = DataLoader(val_dataset_exp1, batch_sampler=val_sampler_exp1, shuffle=False)
test_loader_exp1 = DataLoader(test_dataset_exp1, batch_sampler=test_sampler_exp1, shuffle=False)

# Experience 2: using the filtered DataFrames
train_sampler_exp2 = BalancedBatchSampler(
    data_frame=filtered_train_data_exp2, 
    batch_size=15, 
    samples_per_class=5
)
val_sampler_exp2 = BalancedBatchSampler(
    data_frame=filtered_valid_data_exp2, 
    batch_size=15, 
    samples_per_class=5
)
test_sampler_exp2 = BalancedBatchSampler(
    data_frame=filtered_test_data_exp2, 
    batch_size=15, 
    samples_per_class=5
)

train_loader_exp2 = DataLoader(train_dataset_exp2, batch_sampler=train_sampler_exp2, shuffle=False)
val_loader_exp2 = DataLoader(val_dataset_exp2, batch_sampler=val_sampler_exp2, shuffle=False)
test_loader_exp2 = DataLoader(test_dataset_exp2, batch_sampler=test_sampler_exp2, shuffle=False)

# Experience 3: using the filtered DataFrames
train_sampler_exp3 = BalancedBatchSampler(
    data_frame=filtered_train_data_exp3, 
    batch_size=15, 
    samples_per_class=5
)
val_sampler_exp3 = BalancedBatchSampler(
    data_frame=filtered_valid_data_exp3, 
    batch_size=15, 
    samples_per_class=5
)
test_sampler_exp3 = BalancedBatchSampler(
    data_frame=filtered_test_data_exp3, 
    batch_size=15, 
    samples_per_class=5
)

train_loader_exp3 = DataLoader(train_dataset_exp3, batch_sampler=train_sampler_exp3, shuffle=False)
val_loader_exp3 = DataLoader(val_dataset_exp3, batch_sampler=val_sampler_exp3, shuffle=False)
test_loader_exp3 = DataLoader(test_dataset_exp3, batch_sampler=test_sampler_exp3, shuffle=False)

print("DataLoaders for all experiences created successfully!")


# ## Checking class distribution in each dataset

# In[71]:


import torch
from collections import Counter

def count_classes(dataset):
    # Convert the FlatData into a list of values via list comprehension.
    values = [x for x in dataset.targets]
    # Convert the list of values to a tensor.
    t = torch.tensor(values)
    # Now, convert the tensor to a NumPy array and count the classes.
    return Counter(t.numpy())

print("Class distribution in Train Dataset 1:", count_classes(train_dataset_exp1))
print("Class distribution in Train Dataset 2:", count_classes(train_dataset_exp2))
print("Class distribution in Train Dataset 3:", count_classes(train_dataset_exp3))
print("Class distribution in Validation Dataset 1:", count_classes(val_dataset_exp1))
print("Class distribution in Validation Dataset 2:", count_classes(val_dataset_exp2))
print("Class distribution in Validation Dataset 3:", count_classes(val_dataset_exp3))
print("Class distribution in Test Dataset 1:", count_classes(test_dataset_exp1))
print("Class distribution in Test Dataset 2:", count_classes(test_dataset_exp2))
print("Class distribution in Test Dataset 3:", count_classes(test_dataset_exp3))


# ## Checking unique classes in each experience

# In[ ]:


from avalanche.benchmarks.utils import DataAttribute
from avalanche.benchmarks import benchmark_from_datasets
# Create the benchmark from your datasets
dataset_streams = {
    "train": [train_dataset_exp1, train_dataset_exp2, train_dataset_exp3],
    "test": [test_dataset_exp1, test_dataset_exp2, test_dataset_exp3]
}
# You might want to ensure the benchmark is created here
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


# ## Implementing EWC using Avalanche - the end-to-end continual learning library

# In[ ]:


import os
import csv
import itertools
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.training import EWC
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
from models.cnn_models import SimpleCNN

# -------------------------------
# Create main folder for experiment outputs
# -------------------------------
MAIN_OUT_FOLDER = "ewc_experiment"
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
# Setup loggers and device
# -------------------------------
tb_logger = TensorboardLogger()
text_logger = TextLogger(open('log.txt', 'a'))
interactive_logger = InteractiveLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Setup benchmark and validation datasets
# -------------------------------

dataset_streams = {
    "train": [train_dataset_exp1, train_dataset_exp2, train_dataset_exp3],
    "test": [test_dataset_exp1, test_dataset_exp2, test_dataset_exp3]
}
benchmark = benchmark_from_datasets(**dataset_streams)
# Also store the validation datasets for later use
validation_datasets = [val_dataset_exp1, val_dataset_exp2, val_dataset_exp3]

# -------------------------------
# Grid search loop over hyperparameters.
# -------------------------------
learning_rates = [0.001]
ewc_lambdas = [50, 60, 70, 80, 90, 100]
results_summary = []

# Loop over candidate combinations.
for lr, ewc_lambda in itertools.product(learning_rates, ewc_lambdas):
    print(f"\n=== Hyperparameters: lr={lr}, ewc_lambda={ewc_lambda} ===")
    
    # Create a folder for this hyperparameter configuration.
    config_folder = os.path.join(MAIN_OUT_FOLDER, f"lr{lr}_lambda{ewc_lambda}")
    os.makedirs(config_folder, exist_ok=True)
    
    # Prepare a CSV file for summary metrics for this configuration.
    csv_file_path = os.path.join(config_folder, f"summary_lr{lr}_lambda{ewc_lambda}.csv")
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Experience", "Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc"])
    
    # For each experience.
    for exp_idx, experience in enumerate(benchmark.train_stream):
        print(f"\n=== Start of Experience {experience.current_experience} ===")
        
        # Select the correct DataLoaders for this experience
        if experience.current_experience == 0:
            current_train_loader = train_loader_exp1
            current_val_loader   = val_loader_exp1
            current_test_loader  = test_loader_exp1
        elif experience.current_experience == 1:
            current_train_loader = train_loader_exp2
            current_val_loader   = val_loader_exp2
            current_test_loader  = test_loader_exp2
        elif experience.current_experience == 2:
            current_train_loader = train_loader_exp3
            current_val_loader   = val_loader_exp3
            current_test_loader  = test_loader_exp3
        else:
            raise ValueError("Unexpected experience id")
        
        # Create a folder for this experience.
        exp_folder = os.path.join(config_folder, f"experience_{experience.current_experience}")
        os.makedirs(exp_folder, exist_ok=True)
        
        # Use the corresponding validation DataLoader to create a validation benchmark.
        # (Here, we create a benchmark from the underlying dataset of the validation loader.)
        val_benchmark = benchmark_from_datasets(
            train=[current_val_loader.dataset],
            test=[current_val_loader.dataset]
        )
        
        # Reinitialize model, criterion, and optimizer.
        model = SimpleCNN(num_classes=3).to(device)
        criterion = CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Setup a learning rate scheduler and plugin.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        lr_plugin = LRSchedulerPlugin(scheduler)
        
        evaluator = EvaluationPlugin(
            accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loggers=[interactive_logger, text_logger, tb_logger]
        )
        
        # Instantiate the EWC strategy.
        # We set train_epochs=1 so we can call train() in a loop for each epoch.
        cl_strategy = EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=15,
            train_epochs=1,
            eval_mb_size=15,
            ewc_lambda=ewc_lambda,
            evaluator=evaluator,
            eval_every=-1,  # We'll do our own per-epoch evaluation.
            device=device,
            plugins=[lr_plugin]
        )
        
        # Lists to store per-epoch metrics.
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        num_epochs = 2
        
        # Train for num_epochs using the custom train DataLoader.
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch} for Experience {experience.current_experience} ...")
            train_res = cl_strategy.train(experience, train_loader=current_train_loader)
            epoch_train_loss = train_res.get("Loss_Epoch/train_phase/train_stream", None)
            epoch_train_acc  = train_res.get("Top1_Acc_Epoch/train_phase/train_stream", None)
            train_loss_history.append(epoch_train_loss)
            train_acc_history.append(epoch_train_acc)
            
            # Evaluate on the validation DataLoader.
            val_res = cl_strategy.eval(val_benchmark.test_stream)
            epoch_val_loss = val_res.get("Loss_Stream/eval_phase/test_stream", None)
            epoch_val_acc  = val_res.get("Top1_Acc_Stream/eval_phase/test_stream", None)
            val_loss_history.append(epoch_val_loss)
            val_acc_history.append(epoch_val_acc)
            
            print(f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
            
            # Step the scheduler.
            scheduler.step()
            
            # Log this epoch's metrics.
            log_metrics(csv_file_path, experience.current_experience, epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)
        
        # Plot losses.
        epochs_range = list(range(1, num_epochs + 1))
        loss_title = f"Exp {experience.current_experience}: lr={lr}, ewc_lambda={ewc_lambda} (Loss)"
        loss_plot_path = os.path.join(exp_folder, f"loss_plot_exp{experience.current_experience}.png")
        plot_metrics(epochs_range, train_loss_history, val_loss_history, "Loss", loss_title, loss_plot_path)
        
        # Plot accuracies.
        acc_title = f"Exp {experience.current_experience}: lr={lr}, ewc_lambda={ewc_lambda} (Accuracy)"
        acc_plot_path = os.path.join(exp_folder, f"acc_plot_exp{experience.current_experience}.png")
        plot_metrics(epochs_range, train_acc_history, val_acc_history, "Accuracy", acc_title, acc_plot_path)
        
        # Evaluate on the entire test DataLoader.
        print("Testing on the entire test stream...")
        test_res = cl_strategy.eval(benchmark.test_stream)
        print("Test results:", test_res)
    
    # Optionally, store a summary for this hyperparameter configuration.
    results_summary.append({
        "lr": lr,
        "ewc_lambda": ewc_lambda,
        "final_train_loss": train_loss_history[-1],
        "final_val_loss": val_loss_history[-1],
        "test_results": test_res
    })

print("\n=== Hyperparameter Search Summary ===")
for res in results_summary:
    print(res)


# In[ ]:




