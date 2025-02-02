#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import os
import labels
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm  # Import tqdm for progress visualization
from models.cnn_models import SimpleCNN
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix



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


# ## Create a random temperature sub list

# In[ ]:


# Check if we have enough unique temperatures to select from
if len(unique_temperatures) >= 50:
    # Select the lowest and highest temperatures
    temperature_sublist = [temperature_min, temperature_max]

    # Remove the lowest and highest temperatures from the unique temperatures list
    remaining_temperatures = [temp for temp in unique_temperatures if temp != temperature_min and temp != temperature_max]

    # Randomly select 40 other temperatures from the remaining ones
    random_temperatures = random.sample(remaining_temperatures, 40)

    # Add the random temperatures to the temperature_sublist
    temperature_sublist.extend(random_temperatures)
    
    # Sort from lowest to highest hotend temperature
    temperature_sublist = sorted(temperature_sublist)

    # Print the temperature sublist
    print("\nTemperature sublist:")
    print(temperature_sublist)
else:
    print("Not enough unique temperatures to select from. At least 40 unique temperatures are required.")


# ## Create a new dataframe with equal class distribution

# In[ ]:


# Initialise a dictionary to store DataFrames for each class
class_datasets = {}

# Iterate through the filtered dataset to gather class-wise data
for class_id in [0, 1, 2]:  # Ensure we process all classes: 0, 1, 2
    # Filter the data for the current class
    class_data = data_filtered[data_filtered['hotend_class'] == class_id]
    
    if class_data.empty:
        print(f"Class {class_id} dataset size: 0")
    else:
        # Store the data for each class in the dictionary
        class_datasets[class_id] = class_data
        print(f"Class {class_id} dataset size: {len(class_data)}")

# Find the class with the fewest images
min_class_size = min(len(class_datasets[class_id]) for class_id in class_datasets)

# Print the class with the fewest images
print(f"\nSmallest class size: {min_class_size}")

# Now, we will sample the same number of images from each class
balanced_data = []

# Iterate over each class and sample min_class_size images
for class_id in class_datasets:
    class_data = class_datasets[class_id]
    
    # Randomly sample 'min_class_size' images from the class data
    sampled_class_data = class_data.sample(n=min_class_size, random_state=42)
    balanced_data.append(sampled_class_data)

# Combine all the sampled class data into one DataFrame
balanced_dataset = pd.concat(balanced_data).reset_index(drop=True)

# Display the balanced dataset summary
print(f"\nBalanced dataset size: {len(balanced_dataset)}")

# OPTIONAL: Shuffle the final balanced dataset
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the first and last five rows of the shuffled dataset
print("\nFirst five rows of the shuffled balanced dataset:")
print(balanced_dataset.head())

print("\nLast five rows of the shuffled balanced dataset:")
print(balanced_dataset.tail())

# Print the count of images in each class after balancing
print("\nNumber of images in each hotend class in the balanced dataset:")
for class_id in [0, 1, 2]:
    class_count = len(balanced_dataset[balanced_dataset['hotend_class'] == class_id])
    print(f"Class {class_id}: {class_count} images")


# ## Convert balanced_dataset into a dataframe that contains only the img_path and hotend_class

# In[ ]:


# Assuming the previous steps for balancing the dataset are already done...

# Select only the 'img_path' and 'hotend_class' columns
balanced_dataset_filtered = balanced_dataset[['img_path', 'hotend_class']]

# Display the first few rows of the filtered DataFrame
print("\nFirst five rows of the filtered balanced dataset:")
print(balanced_dataset_filtered.head())

# Display the last few rows of the filtered DataFrame
print("\nLast five rows of the filtered balanced dataset:")
print(balanced_dataset_filtered.tail())

# Optionally, if you want to save this filtered DataFrame to a CSV
#balanced_dataset_filtered.to_csv('balanced_dataset_filtered.csv', index=False)


# In[ ]:


# Check class distribution in balanced_dataset
class_distribution = balanced_dataset_filtered['hotend_class'].value_counts()
print(class_distribution)


# In[ ]:


# Print the indices, the classes, and the number of images in each class
for class_label in class_distribution.index:
    # Get all indices for the current class
    class_indices = balanced_dataset_filtered[balanced_dataset_filtered['hotend_class'] == class_label].index.tolist()
    
    # Count the number of images for the current class
    num_images_in_class = len(class_indices)
    
    # Print the details for this class
    print(f"\nClass: {class_label} (Total images: {num_images_in_class})")
    print("Indices: ", class_indices)
    print(f"Number of images in class {class_label}: {num_images_in_class}")

# Step 1: Get the number of unique classes
num_classes = len(class_distribution)

# Step 2: Set a small batch size
small_batch_size = 15  # You can change this to a value like 32, 64, etc.

# Step 3: Calculate the number of samples per class per batch
samples_per_class = small_batch_size // num_classes  # Ensure it's divisible

# Make sure we don't ask for more samples than available in the smallest class
samples_per_class = min(samples_per_class, class_distribution.min())

# Step 4: Calculate the total batch size
batch_size = samples_per_class * num_classes

print(f"\nRecommended Small Batch Size: {batch_size}")
print(f"Samples per class: {samples_per_class}")


# ## At this point the balanced dataset has been created

# ## Create training, validation, and testing datasets

# In[ ]:


# Number of images in each class (this will be the same after balancing)
num_images_per_class = len(balanced_dataset_filtered) // 3  # since there are 3 classes

# Calculate the number of samples per class
train_size = int(0.8 * num_images_per_class)
valid_size = int(0.1 * num_images_per_class)
test_size = num_images_per_class - train_size - valid_size

# Sample indices for each class
train_indices = []
valid_indices = []
test_indices = []

for class_label in [0, 1, 2]:
    class_data = balanced_dataset_filtered[balanced_dataset_filtered['hotend_class'] == class_label].index.tolist()
    
    # Shuffle the indices of the current class
    random.shuffle(class_data)
    
    # Split the indices for each class into train, validation, and test
    train_indices.extend(class_data[:train_size])
    valid_indices.extend(class_data[train_size:train_size + valid_size])
    test_indices.extend(class_data[train_size + valid_size:])

# Sort the indices of the training, validation, and test datasets to ensure consistent and ordered processing
train_indices = sorted(train_indices)
valid_indices = sorted(valid_indices)
test_indices = sorted(test_indices)

# Class distribution in train, validation, and test sets
train_class_distribution = [0, 0, 0]
valid_class_distribution = [0, 0, 0]
test_class_distribution = [0, 0, 0]

for index in train_indices:
    class_label = balanced_dataset_filtered.loc[index, 'hotend_class']
    train_class_distribution[class_label] += 1

for index in valid_indices:
    class_label = balanced_dataset_filtered.loc[index, 'hotend_class']
    valid_class_distribution[class_label] += 1

for index in test_indices:
    class_label = balanced_dataset_filtered.loc[index, 'hotend_class']
    test_class_distribution[class_label] += 1

# Print the class distribution
print("Train set class distribution:", train_class_distribution)
print("Validation set class distribution:", valid_class_distribution)
print("Test set class distribution:", test_class_distribution)

# Verify lengths
print("Train set size:", len(train_indices))
print("Validation set size:", len(valid_indices))
print("Test set size:", len(test_indices))


# In[ ]:


# Create DataFrames for train, validation, and test sets based on the indices
train_data = balanced_dataset_filtered.iloc[train_indices].reset_index(drop=True)
val_data = balanced_dataset_filtered.iloc[valid_indices].reset_index(drop=True)
test_data = balanced_dataset_filtered.iloc[test_indices].reset_index(drop=True)

# Optionally print the first few rows to verify
print("Train DataFrame sample:")
print(train_data.head())

print("Validation DataFrame sample:")
print(val_data.head())

print("Test DataFrame sample:")
print(test_data.head())


# ## Check for Missing or Invalid Labels in Training, Validation, and Test Data

# In[ ]:


# Check for any missing labels or invalid labels
print(train_data['hotend_class'].isnull().sum())  # Count missing labels
print(train_data['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values

# Check for any missing labels or invalid labels
print(val_data['hotend_class'].isnull().sum())  # Count missing labels
print(val_data['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values
# Check for any missing labels or invalid labels
print(test_data['hotend_class'].isnull().sum())  # Count missing labels
print(test_data['hotend_class'].unique())  # Check unique labels to ensure there are no unexpected values


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


# Create the dataset instance (make sure to provide the right data_frame and root directory)
train_dataset = BalancedDataset(data_frame=train_data, root_dir=root_dir)
val_dataset = BalancedDataset(data_frame=val_data, root_dir=root_dir)
test_dataset = BalancedDataset(data_frame=test_data, root_dir=root_dir)

# Create the sampler (pass the DataFrame instead of the dataset)
train_sampler = BalancedBatchSampler(data_frame=train_data, batch_size=15, samples_per_class=5)
val_sampler = BalancedBatchSampler(data_frame=val_data, batch_size=15, samples_per_class=5)
test_sampler = BalancedBatchSampler(data_frame=test_data, batch_size=15, samples_per_class=5)

# Create the DataLoader with the sampler
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle = False)

# For validation and testing, we typically don't need a batch_sampler, so use regular batching
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, shuffle = False)  # or any batch size that makes sense
test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)  # same as above

print(f"Train dataset length: {len(train_loader.dataset)}")
print(f"Validation dataset length: {len(val_loader.dataset)}")
print(f"Test dataset length: {len(test_loader.dataset)}")


# ## Check the class distribution of randomly selected batches in train loader

# In[ ]:


# Function to print random batches and their class distribution
def print_random_batches(train_loader, num_batches=5):
    for _ in range(num_batches):
        # Get the next batch from the loader
        batch_images, batch_labels, _ = next(iter(train_loader))  # Get the images, labels, and indices (if needed)
        
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
print_random_batches(train_loader, num_batches=5)


# ## Check class distribution of random batches from training, validation and testing data

# In[ ]:


def print_label_batch_from_loader(loader, dataset_name):
    """Fetch and print a batch of labels from the data loader."""
    data_iter = iter(loader)
    batch_images, batch_labels, _ = next(data_iter)  # Get one batch (including the index)
    
    print(f"\n{dataset_name} - Sample Label Batch:")
    print(batch_labels)  # Print the labels for the batch
    
    # Optionally, you can convert the tensor labels to a list for easier reading:
    print(f"Labels as list: {batch_labels.tolist()}")

# Print batches of labels from the train, validation, and test loaders
print_label_batch_from_loader(train_loader, 'Training')
print_label_batch_from_loader(val_loader, 'Validation')
print_label_batch_from_loader(test_loader, 'Test')


# ## Setting up a new folder for each experiment

# In[ ]:


# Automatically create a new experiment folder
base_dir = os.getcwd()  # Change if needed
exp_num = 1

# Find the next available experiment folder
while os.path.exists(os.path.join(base_dir, f"experiment_{exp_num:02d}")):
    exp_num += 1  

experiment_folder = os.path.join(base_dir, f"experiment_{exp_num:02d}")
os.makedirs(experiment_folder)

# Create subdirectories for training, validation, and test confusion matrices
train_folder = os.path.join(experiment_folder, "training_confusion_matrices")
val_folder = os.path.join(experiment_folder, "validation_confusion_matrices")
test_folder = os.path.join(experiment_folder, "test_confusion_matrices")

os.makedirs(train_folder)
os.makedirs(val_folder)
os.makedirs(test_folder)

print(f"Experiment {exp_num:02d} - Saving results to: {experiment_folder}")


# ## Display a Random Image from the Dataset with Its Label

# In[ ]:


import matplotlib.pyplot as plt
import random

# Assume 'train_data' is your DataFrame with image paths and labels
random_index = random.choice(train_data.index)  # Choose a random index
img_path = os.path.join(root_dir, train_data.iloc[random_index, 0])
label = train_data.loc[random_index, 'hotend_class']

# Load the image
img = plt.imread(img_path)  # Use appropriate image loading method

# Plot the image and set the title
plt.imshow(img)
plt.title(f"Label: {label}")

# Define the path to save the image inside the current experiment folder
output_path = os.path.join(experiment_folder, "output_image.png")

# Save the figure in the experiment folder
plt.savefig(output_path)

# Optional: Clear the plot to avoid overlaps in subsequent operations
plt.clf()

print(f"Image saved to: {output_path}")


# In[ ]:


# Ensure that image paths and labels are correctly aligned

# First image
first_index = train_data.index[0]
first_image = train_data.loc[first_index, 'img_path']
first_label = train_data.loc[first_index, 'hotend_class']
print(f"First Image Path: {first_image}, First Label: {first_label}")

# Last image
last_index = train_data.index[-1]  # Accessing the last index
last_image = train_data.loc[last_index, 'img_path']
last_label = train_data.loc[last_index, 'hotend_class']
print(f"Last Image Path: {last_image}, Last Label: {last_label}")


# ## Printing Class Distribution for Training, Validation, and Test Data

# In[ ]:


# Function to print class distribution
def print_class_distribution(loader, dataset_name):
    """Print class distribution in the dataset."""
    all_labels = []

    # Collect all labels from the dataset
    for batch in loader:
        if len(batch) == 2:  # Normal batch with (image, label)
            _, labels = batch
        elif len(batch) == 3:  # Batch with (image, label, idx) from BalancedDataset
            _, labels, _ = batch
        
        # Collect labels from the batch
        all_labels.extend(labels.cpu().numpy())  # Collect labels and move them to CPU if using GPU

    # Calculate and print the class distribution
    class_counts = Counter(all_labels)
    print(f"\n{dataset_name} Class Distribution:")
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} samples")

# Print class distribution for train, validation, and test data
print_class_distribution(train_loader, 'Training')
print_class_distribution(val_loader, 'Validation')
print_class_distribution(test_loader, 'Test')


# ## Model Training, Validation, and Testing with Class Distribution and Learning Rate Scheduling

# In[ ]:


# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = SimpleCNN(num_classes=3).to(device)

# Option to load a pretrained model or start fresh
load_pretrained = False  # Set to False if you want to start with a new model

# Load the model if exists
model_path = "best_model.pth"
if load_pretrained and os.path.exists(model_path):
    print(f"Loading pretrained model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
else:
    print("Starting with a new model...")

# Load best validation accuracy safely
if load_pretrained and os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0) if isinstance(checkpoint, dict) else 0.0
else:
    best_val_accuracy = 0.0

# Training parameters
num_epochs = 100  # Adjust as needed
class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)  # Update these based on your class distribution
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adjust learning rate if needed
# **Add the learning rate scheduler here**
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decrease LR every 10 epochs by a factor of 0.1

# Initialize confusion matrix trackers
num_classes = 3
train_cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
val_cm = ConfusionMatrix(task='multiclass',num_classes=num_classes).to(device)

# Store losses for plotting
train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    class_counts = [0] * 3  # Assuming 3 classes, update if needed

    # Training phase with tqdm progress bar
    for images, labels, _ in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        print(f"Outputs (Raw): {outputs}")  # Log raw outputs
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track training loss and accuracy
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update confusion matrix
        train_cm.update(predicted, labels)

        # Update class counts
        for label in labels:
            class_counts[label.item()] += 1
        
        # Print predicted vs actual labels for each batch
        for i in range(len(labels)):
            print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
    
    # Print class distribution during training
    print(f"Training Class Distribution: {class_counts}")
    
    # **Call the scheduler here at the end of each epoch to update the learning rate**
    scheduler.step()

    # Store training loss for plotting
    train_losses.append(epoch_loss)
    
    # Compute and plot confusion matrix for training
    cm_train = train_cm.compute()
    print(f"Training Confusion Matrix:\n{cm_train}")
    sns.heatmap(cm_train.cpu().numpy(), annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Training Confusion Matrix - Epoch {epoch + 1}')
    output_path_train = os.path.join(train_folder, f"training_confusion_matrix_epoch_{epoch + 1}.png")
    plt.savefig(output_path_train)  # Save the plot
    plt.clf()  # Clear the plot for the next iteration
    print(f"Training Confusion Matrix saved to: {output_path_train}")

    train_cm.reset()  # Reset confusion matrix tracker for next epoch
    print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

    # Validation phase with tqdm progress bar
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    val_class_counts = [0] * 3  # Assuming 3 classes, update if needed

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels, _ in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            print(f"Outputs (Raw): {outputs}")  # Log raw outputs
            loss = criterion(outputs, labels)

            # Track validation loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == labels).sum().item()
            val_total_samples += labels.size(0)
            
            # Update confusion matrix
            val_cm.update(predicted, labels)

            # Update class counts for validation
            for label in labels:
                val_class_counts[label.item()] += 1

            # Print predicted vs actual labels for each batch
            for i in range(len(labels)):
                print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")

    val_epoch_loss = val_loss / val_total_samples
    val_epoch_accuracy = val_correct_predictions / val_total_samples
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")
    
    # Print class distribution during validation
    print(f"Validation Class Distribution: {val_class_counts}")

    # Store validation loss for plotting
    val_losses.append(val_epoch_loss)
    
    # Compute and plot confusion matrix for validation
    cm_val = val_cm.compute()
    print(f"Validation Confusion Matrix:\n{cm_val}")
    sns.heatmap(cm_val.cpu().numpy(), annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Validation Confusion Matrix - Epoch {epoch + 1}')
    output_path_val = os.path.join(val_folder, f"validation_confusion_matrix_epoch_{epoch + 1}.png")
    plt.savefig(output_path_val)  # Save the plot
    plt.clf()  # Clear the plot for the next iteration
    print(f"Validation Confusion Matrix saved to: {output_path_val}")

    val_cm.reset()  # Reset confusion matrix tracker for next epoch
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

    # Save the model if it achieves better validation accuracy
    if val_epoch_accuracy > best_val_accuracy:
        best_val_accuracy = val_epoch_accuracy
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        print("Saved the model with improved validation accuracy.")

# End of training loop
print("Training complete.")

# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.legend()
plt.grid(True)

# Save the plot to the current working directory
output_path = os.path.join(experiment_folder, "training_validation_loss.png")
plt.savefig(output_path)  # Save the plot
plt.clf()  # Clear the plot to free memory for future use
print(f"Training and Validation Loss plot saved to: {output_path}")

# Test model function with tqdm progress bar
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    test_class_counts = [0] * 3  # Assuming 3 classes, update if needed
    all_labels = []
    all_predictions = []
    with torch.no_grad():  # Disable gradients for testing
        for images, labels, _ in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Store labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update class counts for testing
            for label in labels:
                test_class_counts[label.item()] += 1

            # Print predicted vs actual labels for each batch
            for i in range(len(labels)):
                print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")

    avg_accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {avg_accuracy:.4f}")
    
    # Print class distribution during testing
    print(f"Test Class Distribution: {test_class_counts}")
    
     # Generate confusion matrix
    cm_test = confusion_matrix(all_labels, all_predictions, labels=range(3))
    print(f"Test Confusion Matrix:\n{cm_test}")

    # Plot confusion matrix
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(3), yticklabels=range(3))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Confusion Matrix')

    # Save confusion matrix plot
    output_path_test = os.path.join(test_folder, "test_confusion_matrix.png")
    plt.savefig(output_path_test)
    plt.clf()  # Clear the plot for the next use
    print(f"Test Confusion Matrix saved to: {output_path_test}")


# Run the test phase after training
test_model(model, test_loader)


# ## Saving the Training Losses and Validation Losses to a CSV file

# In[39]:


import csv
import os

# Ensure the experiment folder exists
os.makedirs(experiment_folder, exist_ok=True)

# Create the CSV file path (inside the experiment folder)
csv_file_path = os.path.join(experiment_folder, "training_validation_losses.csv")

# Header row with the specified headings
header = ["Epoch", "Training Loss", "Validation Loss"]

# Prepare the data to be saved (epoch-wise losses)
losses_data = []

# Add the header first, regardless of whether the file exists or not
losses_data.append(header)

# Add the epoch losses to the data (train and validation losses for each epoch)
for epoch in range(num_epochs):
    # Safely append the epoch data, ensuring that the lists are not out of range
    # If train_losses or val_losses are shorter than num_epochs, handle the missing data
    train_loss = train_losses[epoch] if epoch < len(train_losses) else None
    val_loss = val_losses[epoch] if epoch < len(val_losses) else None
    losses_data.append([epoch + 1, train_loss, val_loss])

# Check if the file already exists
file_exists = os.path.exists(csv_file_path)

# Open the file in write mode to overwrite existing content or create a new file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # If the file exists, we write the header and the new data (clearing existing content first)
    if file_exists:
        # Write the header and the new epoch-wise loss data
        writer.writerows(losses_data)
    else:
        # If the file doesn't exist, just write the new data with the header
        writer.writerow(header)
        writer.writerows(losses_data)

print(f"Training and Validation Losses saved to: {csv_file_path}")


# In[ ]:




