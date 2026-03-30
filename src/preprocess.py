import os
import pandas as pd

# ---- STEP 1: Path to your XPT file ----
xpt_path = os.path.expanduser("~/Downloads/LLCP2024.XPT")

# ---- STEP 2: Where to save your smaller dataset ----
output_path = os.path.join("data", "brfss_small.csv")

# ---- STEP 3: Select relevant features ----
selected_cols = [
    "_MENT14D",   # target

    # lifestyle
    "EXERANY2",
    "SMOKE100",
    "SMOKDAY2",
    "ALCDAY4",
    "DRNK3GE5",

    # health
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",

    # socioeconomic
    "INCOME3",
    "EDUCA",
    "EMPLOY1",
    "MARITAL",

    # social
    "EMTSUPRT",
    "SDLONELY",

    # demographics
    "_AGE80",
    "_SEX",
    "_RACE"]

# ---- STEP 4: Define missing value codes ----
missing_codes = [7, 8, 9, 77, 88, 99, 777, 888, 999]

# ---- STEP 5: Read file in chunks ----
reader = pd.read_sas(xpt_path, chunksize=50000)

chunks = []

for chunk in reader:
    # keep only selected columns
    chunk = chunk[selected_cols].copy()

    # replace BRFSS missing codes with NaN
    chunk = chunk.replace(missing_codes, pd.NA)

    # drop rows where target is missing
    chunk = chunk.dropna(subset=["_MENT14D"])

    chunks.append(chunk)

# ---- STEP 6: Combine all chunks ----
df = pd.concat(chunks, ignore_index=True)

# ---- STEP 7: Check result ----
print("Final dataset shape:", df.shape)
print(df.head())

# ---- STEP 8: Save to CSV ----
df.to_csv(output_path, index=False)

print(f"Saved cleaned dataset to: {output_path}")