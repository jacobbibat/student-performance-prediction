import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../studentData.csv", sep=";")

# Create folder for bar charts
os.makedirs("bar_charts", exist_ok=True)

attributes = [
    "Label",
    "Discipline",
    "LearningStyle",
    "Course"
]

for attribute in attributes:

    # Count students in each category
    counts = df[attribute].value_counts()

    # Calculate percentages
    percentages = df[attribute].value_counts(normalize=True) * 100

    # Print results
    print(f"\n{'=' * 40}")
    print(f"Distribution of {attribute}")
    print(f"{'=' * 40}")

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages.round(2)
    })

    print(summary.to_string())

    # Create bar chart
    plt.figure(figsize=(10, 5))

    counts.plot(kind="bar", edgecolor="black")

    plt.title(f"Distribution of {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save bar chart
    plt.savefig(
        f"bar_charts/{attribute}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()