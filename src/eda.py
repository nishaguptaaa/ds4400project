import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading data
base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, "data", "brfss_small.csv")
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)
print(df["_MENT14D"].value_counts(dropna=False))
print()

#keeping only valid binary values
df = df[df["_MENT14D"].isin([1.0, 2.0])].copy()

print("Filtered dataset shape:", df.shape)
print(df["_MENT14D"].value_counts(dropna=False))

#recoding variables for readability
df["_MENT14D_label"] = df["_MENT14D"].map({1.0: "Yes", 2.0: "No"})
df["EXERANY2_label"] = df["EXERANY2"].map({1.0: "Yes", 2.0: "No"})

#plot style
sns.set_theme(style="whitegrid")
palette = {"Yes": "#ff69b4", "No": "#9370db"}

#Plot 1: Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="_MENT14D_label",
    hue="_MENT14D_label",
    hue_order=["Yes", "No"],
    palette=palette,
    legend=False)
plt.title("Distribution of Frequent Mental Distress", fontsize=14, fontweight="bold")
plt.xlabel("Frequent Mental Distress")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


#Plot 2: Exercise vs mental distress
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=df,
    x="EXERANY2_label",
    hue="_MENT14D_label",
    hue_order=["Yes", "No"],
    palette=palette)
plt.title("Frequent Mental Distress by Exercise Status", fontsize=14, fontweight="bold")
plt.xlabel("Exercised in Past Month")
plt.ylabel("Count")
ax.legend(title="Frequent Mental Distress")
plt.tight_layout()
plt.show()


#Plot 3: Loneliness vs mental distress
#loneliness labels
df["SDLONELY_label"] = df["SDLONELY"].map({
    1.0: "Always",
    2.0: "Usually",
    3.0: "Sometimes",
    4.0: "Rarely",
    5.0: "Never"})

plt.figure(figsize=(6, 4))
ax = sns.boxplot(
    data=df,
    x="_MENT14D_label",
    y="SDLONELY",
    hue="_MENT14D_label",
    hue_order=["Yes", "No"],
    palette=palette,
    dodge=False)
plt.title("Loneliness and Frequent Mental Distress", fontsize=14, fontweight="bold")
plt.xlabel("Frequent Mental Distress")
plt.ylabel("Loneliness (1 = Always, 5 = Never)")
if ax.legend_ is not None:
    ax.legend_.remove()
plt.tight_layout()
plt.show()


#Plot 4: Income vs mental distress
#income labels
df["INCOME3_label"] = df["INCOME3"].map({
    1.0: "<$10k",
    2.0: "$10k-$15k",
    3.0: "$15k-$20k",
    4.0: "$20k-$25k",
    5.0: "$25k-$35k",
    6.0: "$35k-$50k",
    10.0: "$50k-$100k",
    11.0: "$100k+"})

income_order = ["<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k",
                "$25k-$35k", "$35k-$50k", "$50k-$100k", "$100k+"]

plt.figure(figsize=(9, 4))
ax = sns.countplot(
    data=df,
    x="INCOME3_label",
    order=income_order,
    hue="_MENT14D_label",
    hue_order=["Yes", "No"],
    palette=palette)
plt.title("Frequent Mental Distress by Income Group", fontsize=14, fontweight="bold")
plt.xlabel("Income Group")
plt.ylabel("Count")
plt.xticks(rotation=30)
ax.legend(title="Frequent Mental Distress")
plt.tight_layout()
plt.show()


#Plot 5: General health vs mental distress
#general health labels
df["GENHLTH_label"] = df["GENHLTH"].map({
    1.0: "Excellent",
    2.0: "Very good",
    3.0: "Good",
    4.0: "Fair",
    5.0: "Poor"})

health_order = ["Excellent", "Very good", "Good", "Fair", "Poor"]

plt.figure(figsize=(8, 4))
ax = sns.countplot(
    data=df,
    x="GENHLTH_label",
    order=health_order,
    hue="_MENT14D_label",
    hue_order=["Yes", "No"],
    palette=palette)
plt.title("Frequent Mental Distress by General Health", fontsize=14, fontweight="bold")
plt.xlabel("General Health")
plt.ylabel("Count")
ax.legend(title="Frequent Mental Distress")
plt.tight_layout()
plt.show()

#summaries of plots
print("\nTarget distribution:")
print(df["_MENT14D_label"].value_counts(normalize=True))

print("\nExercise vs distress:")
print(pd.crosstab(df["EXERANY2_label"], df["_MENT14D_label"], normalize="index"))

print("\nIncome vs distress:")
print(pd.crosstab(df["INCOME3_label"], df["_MENT14D_label"], normalize="index"))

print("\nGeneral health vs distress:")
print(pd.crosstab(df["GENHLTH_label"], df["_MENT14D_label"], normalize="index"))