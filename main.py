import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Download the latest version of the Iris dataset
path = kagglehub.dataset_download("uciml/iris")

print("Path to dataset files:", path)

# Dataset file path
iris_data_path = os.path.join(path, "Iris.csv")

print(f"Loading Iris dataset from: {iris_data_path}")

# Check if the dataset exists
if not os.path.exists(iris_data_path):
    print(f"Error: Iris dataset file not found at {iris_data_path}")
else:
    # Load the dataset
    iris_df = pd.read_csv(iris_data_path)

    # Features and target
    X = iris_df.drop(["Id", "Species"], axis=1)
    y = iris_df["Species"]

    print("\nIris dataset loaded successfully.")
    print(f"Number of samples: {len(iris_df)}")

    print("\nFeatures (X):")
    print(X.head())

    print("\nTarget (y):")
    print(y.head())

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    print("\nTraining RandomForestClassifier...")

    # Create and train model
    rf_iris_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf_iris_model.fit(X_train, y_train)

    print("\nMaking predictions...")

    # Predict
    y_pred = rf_iris_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
