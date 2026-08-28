"""Load all six benchmark CSVs and print a basic profile of each."""

import pandas as pd

FILES = [
    "01_freMTPL2_freq.csv",
    "02_freMTPL2_sev.csv",
    "03_wasa_motorcycle.csv",
    "04_ausprivauto_dejong.csv",
    "05_eudirectlapse_demand.csv",
]

for filename in FILES:
    df = pd.read_csv(filename)

    print("=" * 100)
    print(f"FILE: {filename}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nColumn dtypes:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nFirst 5 rows:")
    print(df.head(5).to_string())
    print()
