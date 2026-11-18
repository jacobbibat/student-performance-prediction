import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../studentData.csv", sep=";")

# Create folder for correlation charts
os.makedirs("correlation_charts", exist_ok=True)

numeric_attributes = [
    "ExtrinsicMotivation",
    "GroupWork",
    "IntrinsicMotivation",
    "SelfEfficacy",
    "SelfRegulation",
    "StudyEffort",
    "StudyTime",
    "Openness",
    "Conscientiousness",
    "HighSchoolAverage",
    "HighSchoolEnglish",
    "HighSchoolMaths",
    "Age"
]

# Calculate Pearson correlations
correlation_matrix = df[numeric_attributes].corr()

# Create heatmap
plt.figure(figsize=(13, 10))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Pearson Correlation")

plt.xticks(
    range(len(numeric_attributes)),
    numeric_attributes,
    rotation=90
)

plt.yticks(
    range(len(numeric_attributes)),
    numeric_attributes
)

# Display correlation values inside each square
for i in range(len(numeric_attributes)):
    for j in range(len(numeric_attributes)):

        value = correlation_matrix.iloc[i, j]

        plt.text(
            j, i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="black"
        )

plt.title("Correlation Heatmap of Student Attributes")
plt.tight_layout()

# Save heatmap
plt.savefig(
    "correlation_charts/CorrelationHeatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()