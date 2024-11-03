import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=0, dtype=str)  # Read CSV with headers
        self.root_dir = root_dir
        self.transform = transform or self.default_transform()  # Use default transform if none provided

        # Filter the dataset to include only valid entries
        self.valid_indices = self.get_valid_indices()

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert to tensor
        ])

    def get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx, 0].strip()  # Ensure no extra spaces
            img_name = img_name.split('/')[-1]  # Keep only the image file name

            # Construct full image path
            full_img_path = os.path.join(self.root_dir, img_name)

            # Check if the image exists directly in the specified image folder
            if os.path.exists(full_img_path):
                valid_indices.append(idx)  # Only include valid indices
            else:
                print(f"Image does not exist: {full_img_path}")  # Log missing images

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]  # Get the actual index for valid data
        img_name = self.data.iloc[actual_idx, 0].strip()  # Keep only the image file name
        full_img_path = os.path.join(self.root_dir, img_name)  # Ensure this is correct

        # Get the hotend class from the 16th column (index 15)
        hotend_class = self.data.iloc[actual_idx, 15].strip()  # Remove spaces

        try:
            hotend_class = int(hotend_class)  # Convert to integer
        except ValueError:
            print(f"Invalid hotend class for image {img_name}. Skipping...")
            return self.__getitem__((idx + 1) % len(self.valid_indices))  # Loop to find the next valid item

        try:
            image = Image.open(full_img_path).convert('RGB')  # Open image
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.valid_indices))  # Loop to find the next valid item

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, hotend_class  # Return the image and hotend class


class BalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=15):
        self.data_source = data_source
        self.batch_size = batch_size

        # Ensure the batch size is evenly divisible by the number of classes
        self.num_classes = len(set(label for _, label in data_source))  # Get number of unique classes
        if self.batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")

        self.samples_per_class = self.batch_size // self.num_classes

        # Group indices by class
        self.class_indices = {
            class_id: np.array([idx for idx, (_, label) in enumerate(data_source) if label == class_id])
            for class_id in set(label for _, label in data_source)
        }

        # Shuffle class indices
        for class_id in self.class_indices:
            np.random.shuffle(self.class_indices[class_id])

    def __len__(self):
        # Calculate the total number of batches
        min_class_samples = min(len(indices) for indices in self.class_indices.values())
        return min_class_samples // self.samples_per_class

    def __iter__(self):
        while True:
            batch = []
            for class_id, indices in self.class_indices.items():
                if len(indices) < self.samples_per_class:
                    return  # Stop iteration if any class doesn't have enough samples

                # Take samples for the batch from each class
                batch.extend(indices[:self.samples_per_class])
                # Remove these samples from the class indices
                self.class_indices[class_id] = indices[self.samples_per_class:]

            # Shuffle within the batch
            np.random.shuffle(batch)
            yield batch  # Yield the current batch


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the dataset into training, validation, and testing sets.

    :param data: The complete dataset (DataFrame).
    :param train_ratio: Proportion of data to use for training.
    :param val_ratio: Proportion of data to use for validation.
    :return: A tuple of DataFrames for train, validation, and test.
    """
    # Calculate sizes
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split the dataset using the indices
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return data.iloc[train_indices], data.iloc[val_indices], data.iloc[test_indices]


def create_dataloader(csv_file, root_dir, batch_size, transform):
    # Create an instance of the CustomDataset
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # Create the DataLoader with the balanced sampler
    balanced_sampler = BalancedBatchSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=balanced_sampler)

    # Debug: Print the total number of balanced batches possible
    print(f'Total balanced batches possible: {len(balanced_sampler)}')
    print(f'Dataloader size (number of batches): {len(dataloader)}')

    return dataloader


# Paths to your files
csv_file = r'C:\Users\Sandhra George\avalanche\data\dataset.csv'  # Path to the CSV file
root_dir = r'C:\Users\Sandhra George\avalanche\caxton_dataset\print0'  # Path to the image directory

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Load the dataset
balanced_data = pd.read_csv(csv_file, header=0, dtype=str)  # Load the dataset first

# Split the dataset into train, validation, and test
train_data, val_data, test_data = split_data(balanced_data)

# Create DataLoaders for each dataset
batch_size = 15
train_dataloader = create_dataloader(csv_file, root_dir, batch_size, transform)

# Create validation and test DataLoaders (without balanced sampling)
val_dataloader = DataLoader(CustomDataset(csv_file, root_dir, transform), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(CustomDataset(csv_file, root_dir, transform), batch_size=batch_size, shuffle=True)

# Now you can iterate through each DataLoader
print("Training batches:")
for images, labels in train_dataloader:
    print(f'Batch shapes - Images: {images.shape}, Labels: {labels}')

print("\nValidation batches:")
for images, labels in val_dataloader:
    print(f'Batch shapes - Images: {images.shape}, Labels: {labels}')

print("\nTesting batches:")
for images, labels in test_dataloader:
    print(f'Batch shapes - Images: {images.shape}, Labels: {labels}')
