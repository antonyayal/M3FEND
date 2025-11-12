import pandas as pd

# Read pkl file
import os

# Use environment variable or relative path for portability
path = os.environ.get("DATA_PATH", "test.pkl")
df = pd.read_pickle(path, compression="infer")  # 'infer' detecta si viene gz/bz2
print(type(df))
print(df.shape)
print(df.head())

#exportar pkl a csv
df.to_csv("test_dataset.csv", index=False)

#convertir csv a pkl
df_csv = pd.read_csv("test_dataset.csv")
df_csv.to_pickle("test_dataset_from_csv.pkl", compression="infer")
print("Conversion from CSV to PKL completed.")
print(type(df_csv))
