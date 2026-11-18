import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../studentData.csv", sep=";")

# Create a folder for the diagrams
os.makedirs("histograms", exist_ok=True)

attributes = [
    "Age",
    "StudyTime",
    "StudyEffort",
    "HighSchoolAverage",
    "IntrinsicMotivation"
]

for attribute in attributes:
    plt.figure(figsize=(7, 4))

    plt.hist(
        df[attribute].dropna(),
        bins=10,
        edgecolor="black"
    )

    plt.title(f"Distribution of {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Number of Students")

    # Save the histogram into the folder
    plt.savefig(f"histograms/{attribute}.png", dpi=300, bbox_inches="tight")

    plt.close()