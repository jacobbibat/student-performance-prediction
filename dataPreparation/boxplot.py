import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../studentData.csv", sep=";")

# Create folder for boxplots
os.makedirs("boxplots", exist_ok=True)

attributes = [
    "Age",
    "SelfEfficacy",
    "StudyTime",
    "HighSchoolAverage"
]

for attribute in attributes:
    plt.figure(figsize=(7, 4))

    plt.boxplot(df[attribute].dropna())

    plt.title(f"Boxplot of {attribute}")
    plt.ylabel(attribute)

    # Save each boxplot
    plt.savefig(
        f"boxplots/{attribute}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()