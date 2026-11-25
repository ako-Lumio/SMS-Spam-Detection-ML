
import pandas as pd

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
data = data[["v1", "v2"]]

# Rename columns
data.columns = ["label", "message"]

# Remove missing and duplicate records
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

print("Dataset Preview:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMessage Distribution:")
print(data["label"].value_counts())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDataset Cleaning Completed!")
