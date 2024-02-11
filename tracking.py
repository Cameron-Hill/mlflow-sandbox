import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("http://localhost:5000")

# Load dataset
print("Loading iris dataset")
X, y = datasets.load_iris(return_X_y=True)

# Split dataset
print("Splitting dataset")
X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define Model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train model
print("Training model")
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test data
print("Predicting on test data")
y_pred = lr.predict(x_test)

# Calculate metric
print("Calculating metrics")
accuracy = accuracy_score(y_test, y_pred)


# Create an MLflow experiment
experiment_name = "MLflow Quick Start"
print(f"Creating an experiment with name {experiment_name}")
mlflow.set_experiment(experiment_name)

# Start an MLFlow run
with mlflow.start_run() as run:
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

print("Done.")
print(f"Model URI: {model_info.model_uri}")
