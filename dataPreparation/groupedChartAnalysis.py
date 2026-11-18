import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../studentData.csv", sep=";")

# Create folder for charts
os.makedirs("pass_fail_charts", exist_ok=True)

attributes = [
    "Discipline",
    "LearningStyle",
    "Course"
]

for attribute in attributes:

    # Exclude missing category values
    data = df.dropna(subset=[attribute, "Label"])

    # Count Pass and Fail for each category
    counts = pd.crosstab(data[attribute], data["Label"])

    # Convert counts into percentages
    percentages = counts.div(
        counts.sum(axis=1), axis=0
    ) * 100

    # Print results in terminal
    print(f"\n{'=' * 50}")
    print(f"Pass/Fail Analysis by {attribute}")
    print(f"{'=' * 50}")

    print("\nStudent Counts:")
    print(counts.to_string())

    print("\nPercentages (%):")
    print(percentages.round(2).to_string())

    # Create grouped bar chart
    percentages.plot(
        kind="bar",
        figsize=(11, 5),
        edgecolor="black"
    )

    plt.title(f"Pass/Fail Percentage by {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Percentage of Students (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Result")
    plt.tight_layout()

    # Save chart
    plt.savefig(
        f"pass_fail_charts/{attribute}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()