import pandas as pd

df = pd.read_csv("data/data.csv")

print("Number of rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStrength distribution:")
print(df["strength"].value_counts())

print("\nStrength percentage distribution:")
print(df["strength"].value_counts(normalize=True) * 100)