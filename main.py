from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Iris Species Prediction API")

def train_model():
    # Load the dataset from the CSV file.
    df = pd.read_csv('iris.csv')

    # Define features (X) and target (y)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'

    X = df[features]
    y = df[target]

    # Since the target is text, we need to encode it into numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y_encoded)

    # Save the trained model and the label encoder
    joblib.dump(model, 'iris_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')

# Train the model on startup
train_model()

# Load the model and encoder for predictions
model = joblib.load('iris_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API. Using data from iris.csv."}

@app.post("/predict")
def predict_species(iris_features: IrisFeatures):
    data = np.array([[
        iris_features.sepal_length,
        iris_features.sepal_width,
        iris_features.petal_length,
        iris_features.petal_width
    ]])

    # Predict the encoded label
    prediction_encoded = model.predict(data)

    # Convert the encoded label back to the species name
    species_name = label_encoder.inverse_transform(prediction_encoded)

    probability = model.predict_proba(data).max()

    return {
        "predicted_species": species_name[0],
        "prediction_probability": float(probability)
    }
