
import pandas as pd

df = pd.read_csv("../studentData.csv", sep=";")

# 1. Dataset dimensions
print("Dataset shape:", df.shape)

# 2. Target distribution
print("\nPass/Fail distribution:")
print(df["Label"].value_counts())
print(df["Label"].value_counts(normalize=True) * 100)

# 3. Missing values
missing = pd.DataFrame({
    "Count": df.isna().sum(),
    "Percentage": df.isna().mean() * 100
})

print("\nMissing values:")
print(missing[missing["Count"] > 0].round(2))

# 4. Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df["id"].duplicated().sum())

# 5. Invalid SelfEfficacy values
invalid = df[
    df["SelfEfficacy"].notna()
    & ~df["SelfEfficacy"].between(0, 10)
]

print("\nInvalid SelfEfficacy values:")
print(invalid[["id", "SelfEfficacy"]])

# 6. Pass/Fail percentages by discipline
print("\nPass/Fail by Discipline:")

discipline_results = pd.crosstab(
    df["Discipline"],
    df["Label"],
    normalize="index"
) * 100

print(discipline_results.round(1))
