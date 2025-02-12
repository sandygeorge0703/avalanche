import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


def display_sequential_images_per_temperature(data, base_path, print_id="print0", random_state=42):
    """
    Displays one image per class for each unique target hotend temperature in sequence.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing image paths, target hotend temperatures, and class information.
    - base_path (str): The absolute path to the directory containing images.
    - print_id (str): Identifier to filter rows by image path (default is "print0").
    - random_state (int): Seed for reproducibility of random image selection (default is 42).
    """
    # Filter data to include only entries with the specified print_id in the first column
    filtered_data = data[data.iloc[:, 0].str.contains(print_id, case=False, na=False)]

    # Check if any data matches the specified print_id
    if filtered_data.empty:
        print(f"No data found for images containing '{print_id}' in the path.")
        return
    else:
        print(f"Filtered data for images with '{print_id}' in path:\n", filtered_data.head())

    # Sort unique target hotend temperatures
    unique_temperatures = sorted(filtered_data['target_hotend'].unique())

    # Print sorted temperatures and their count
    print("\nUnique target hotend temperatures in the dataset (sorted):")
    print(unique_temperatures)
    print(f"Number of unique target hotend temperatures: {len(unique_temperatures)}")

    # Loop through each unique temperature in order
    for temp in unique_temperatures:
        print(f"\nDisplaying one image per class at target hotend temperature {temp}°:")

        # Filter data for the current temperature
        temp_filtered = filtered_data[filtered_data['target_hotend'] == temp]

        # Loop through each class (0, 1, and 2) for the current temperature
        for hotend_class in [0, 1, 2]:
            # Filter data for the current class at this temperature
            class_data = temp_filtered[temp_filtered['hotend_class'] == hotend_class]

            # Check if there's data for this class at the current temperature
            if not class_data.empty:
                # Randomly select one image for the current class and temperature
                random_image = class_data.sample(n=1, random_state=random_state).iloc[0]

                img_filename = random_image.iloc[0].strip()  # Get the image filename from the first column and remove any whitespace
                img_path = os.path.join(base_path, img_filename)  # Construct the full path

                # Debug print statement for constructed path
                print(f"Constructed path: {img_path}")

                # Check if the image file exists
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    plt.figure()
                    plt.imshow(img)
                    plt.title(f"Target Hotend Temperature: {temp}° - Class: {hotend_class}")
                    plt.axis('off')
                    plt.show()
                else:
                    print(f"Image not found at path: {img_path}")
            else:
                print(f"No data found for Class {hotend_class} at Temperature {temp}°.")


# # Usage example
data = pd.read_csv('dataset.csv')  # Load the dataset
base_path = r"C:\Users\Sandhra George\avalanche"  # Define the image directory
display_sequential_images_per_temperature(data, base_path)  # Call the function
