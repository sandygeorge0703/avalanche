import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torch  # Ensure that torch is imported

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=0, dtype=str)  # Read CSV with headers
        self.root_dir = root_dir
        self.transform = transform or self.default_transform()  # Use default transform if none provided

        # Set the image folder directly to the root directory
        self.image_folder = self.root_dir  # Use root_dir directly

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

            # Extract the image number from the filename
            if img_name.startswith("image-"):
                try:
                    image_number = int(img_name.split('-')[1].split('.')[0])
                    if image_number < 3085:
                        # Here, we assume img_name is already just the filename
                        # Check if the image exists directly in the specified image folder
                        if os.path.exists(os.path.join(self.root_dir, img_name)):
                            valid_indices.append(idx)  # Only include valid indices
                        else:
                            print(f"Image does not exist: {img_name}")  # Log missing images
                except ValueError:
                    print(f"Invalid filename format for {img_name}. Skipping...")

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):  # Fixed syntax error here
        if isinstance(idx, list):
            # Handle batch indexing
            images = []
            labels = []
            for i in idx:
                actual_idx = self.valid_indices[i]  # Get the actual index for valid data
                img_name = self.data.iloc[actual_idx, 0].strip()  # Keep only the image file name
                full_img_path = os.path.join(self.image_folder, img_name)  # Ensure this is correct

                # Get the hotend class from the 16th column (index 15)
                hotend_class = self.data.iloc[actual_idx, 15].strip()  # Remove spaces

                try:
                    hotend_class = int(hotend_class)  # Convert to integer
                except ValueError:
                    print(f"Invalid hotend class for image {img_name}. Skipping...")
                    continue  # Skip this item

                try:
                    image = Image.open(full_img_path).convert('RGB')  # Open image
                except Exception as e:
                    print(f"Error loading image {full_img_path}: {e}")
                    continue  # Skip this item

                if self.transform:
                    image = self.transform(image)  # Apply transformations

                images.append(image)
                labels.append(hotend_class)

            return torch.stack(images), torch.tensor(labels)  # Return a batch of images and labels
        else:
            actual_idx = self.valid_indices[idx]  # Get the actual index for valid data
            img_name = self.data.iloc[actual_idx, 0].strip()  # Keep only the image file name
            full_img_path = os.path.join(self.image_folder, img_name)  # Ensure this is correct

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


# Paths to your files
csv_file = r'/data/dataset.csv'  # Path to the CSV file
root_dir = r'/caxton_dataset/print0'  # Path to the image directory

# Example usage
# dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir)
