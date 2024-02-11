import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

MODEL_URL = "runs:/f952c8c1a6fb43c7baab3f642d3c9cd2/iris_model"

mlflow.set_tracking_uri("http://localhost:5000")
# Load dataset
print("Loading iris dataset")
X, y = datasets.load_iris(return_X_y=True)

# Split dataset
print("Splitting dataset")
X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the model
print("Loading the model")
loaded_model = mlflow.pyfunc.load_model(MODEL_URL)


# Predict on the test data
print("Predicting on test data")
predictions = loaded_model.predict(x_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(x_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(f"Result:")
print(result.head(10))
