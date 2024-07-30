from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

# Memuat model dan scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

# Mendefinisikan model data input menggunakan Pydantic
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    # Mengonversi input data ke dalam bentuk array
    data = np.array([[input_data.pregnancies, input_data.glucose, input_data.blood_pressure,
                      input_data.skin_thickness, input_data.insulin, input_data.bmi,
                      input_data.diabetes_pedigree_function, input_data.age]])

    # Standarisasi data
    data = scaler.transform(data)

    # Membuat prediksi
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    # Mengembalikan hasil prediksi
    return {
        "prediction": int(prediction[0]),
        "probability": probability[0].tolist()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug")