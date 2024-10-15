import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Load the CSV file treating the first row as header
        self.data = pd.read_csv(csv_file, header=0, dtype=str)  # Read CSV with headers
        self.root_dir = root_dir
        self.transform = transform or self.default_transform()  # Use default transform if none provided

        # Filter the dataset to include only valid entries (existence of image files and filename check)
        self.valid_indices = self.get_valid_indices()

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),  # Resize to 128x128
            transforms.ToTensor(),  # Convert to tensor
        ])

    def get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx, 0]  # Access image path from the CSV
            img_name = img_name.split('/')[-1]  # Keep only the image file name, discard other paths

            # Extract the image number from the filename (assuming it follows "image-<number>.jpg" format)
            if img_name.startswith("image-"):
                try:
                    image_number = int(img_name.split('-')[1].split('.')[0])  # Extract number from filename
                except ValueError:
                    print(f"Invalid filename format for {img_name}. Skipping...")
                    continue  # Skip if filename is invalid

                # Only process images with number less than 3084
                if image_number < 3084:
                    # Construct full image path in the print0 directory
                    full_img_path = os.path.join(self.root_dir, img_name)

                    # Check if the image exists in the specified directory
                    if os.path.exists(full_img_path):
                        valid_indices.append(idx)  # Only include valid indices where the image exists
                    else:
                        print(f"Image does not exist: {full_img_path}")  # Log missing images
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]  # Get the actual index for valid data
        img_name = self.data.iloc[actual_idx, 0]  # First column is the image path
        img_name = img_name.split('/')[-1]  # Keep only the image file name, discard other paths
        full_img_path = os.path.join(self.root_dir, img_name)  # Construct full image path

        # Get the flow rate class from the 13th column (index 12)
        flow_rate_class = self.data.iloc[actual_idx, 12]  # 13th column should contain flow rate class

        # Convert flow rate class to an integer
        try:
            flow_rate_class = int(flow_rate_class)  # Ensure this is treated as an integer for classification
        except ValueError:
            print(f"Invalid flow rate class for image {img_name}. Skipping...")
            return self.__getitem__((idx + 1) % len(self.valid_indices))  # Recursively try to get the next valid item

        # Load the image
        try:
            image = Image.open(full_img_path).convert('RGB')  # Open image
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.valid_indices))  # Recursively try to get the next valid item

        if self.transform:
            image = self.transform(image)  # Apply transformations

        # Debugging print statements
        print(f"Loaded image: {img_name}, Flow Rate Class: {flow_rate_class}")

        # Ensure image shape is as expected
        if image.size() != (3, 128, 128):
            print(f"Warning: Image shape {image.size()} is not (3, 128, 128) for {img_name}.")

        return image, flow_rate_class  # Return the image and flow rate class
