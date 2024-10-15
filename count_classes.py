import pandas as pd

# Load the CSV file
df = pd.read_csv('data/dataset.csv')  # Update with the actual path to your dataset CSV

# Check how many unique classes are in the 'flow_rate_class' column
unique_classes = df['flow_rate_class'].nunique()

print(f"Number of unique flow_rate_class labels: {unique_classes}")

# Optionally, print the unique class values
print("Unique classes in flow_rate_class:", df['flow_rate_class'].unique())
