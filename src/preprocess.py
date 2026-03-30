import os
import pandas as pd

#loading huge dataset
xpt_path = os.path.expanduser("~/Downloads/LLCP2024.XPT")

#saving smaller data set
output_path = os.path.join("data", "brfss_small.csv")

#selecting relevant features
selected_cols = [
    #target variable
    "_MENT14D",

    #lifestyle
    "EXERANY2",
    "SMOKE100",
    "SMOKDAY2",
    "ALCDAY4",
    "DRNK3GE5",

    #health
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",

    #socioeconomic
    "INCOME3",
    "EDUCA",
    "EMPLOY1",
    "MARITAL",

    #social
    "EMTSUPRT",
    "SDLONELY",

    #demographics
    "_AGE80",
    "_SEX",
    "_RACE"]

#missing value codes
missing_codes = [7, 8, 9, 77, 88, 99, 777, 888, 999]

#reading file in chunks
reader = pd.read_sas(xpt_path, chunksize=50000)
chunks = []

for chunk in reader:
    #keeping only selected columns
    chunk = chunk[selected_cols].copy()

    #replacing missing codes with NaN
    chunk = chunk.replace(missing_codes, pd.NA)

    #dropping rows where target is missing
    chunk = chunk.dropna(subset=["_MENT14D"])

    chunks.append(chunk)

#combining all chunks
df = pd.concat(chunks, ignore_index=True)

#checking results
print("Final dataset shape:", df.shape)
print(df.head())

#saving smaller csv file
df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to: {output_path}")