
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)

# Load the dataset
df = pd.read_csv("studentData.csv", sep=";")

# Separate predictors and target
X = df.drop(columns=["Label", "id"])
y = df["Label"]

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude="number"
).columns.tolist()

# Prepare numerical attributes
numeric_transformer = SimpleImputer(
    strategy="median"
)

# Prepare categorical attributes
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_columns),
    ("categorical", categorical_transformer, categorical_columns)
])

# Create baseline pipeline
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Define evaluation metrics
metrics = {
    "accuracy": "accuracy",
    "precision_fail": make_scorer(
        precision_score, pos_label="Fail"
    ),
    "recall_fail": make_scorer(
        recall_score, pos_label="Fail"
    ),
    "f1_fail": make_scorer(
        f1_score, pos_label="Fail"
    )
}

# Five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = cross_validate(
    baseline,
    X_train,
    y_train,
    cv=cv,
    scoring=metrics
)

# Display average performance
for metric in metrics:
    score = results[f"test_{metric}"].mean()
    print(f"{metric}: {score:.3f}")

# Fit baseline on the complete training set
baseline.fit(X_train, y_train)
