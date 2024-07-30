from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

# Memuat model dan scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Definisikan model input menggunakan Pydantic
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

# Endpoint untuk prediksi
@app.post("/predict")
async def predict(data: DiabetesInput):
    try:
        # Konversi input data ke numpy array
        input_array = np.array([[data.pregnancies, data.glucose, data.blood_pressure,
                                 data.skin_thickness, data.insulin, data.bmi,
                                 data.diabetes_pedigree_function, data.age]])

        # Standarisasi data
        input_array = scaler.transform(input_array)

        # Buat prediksi
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)

        predicted_class = int(prediction[0])
        predicted_probability = (probability[0][predicted_class] * 100)

        # Mengembalikan hasil prediksi
        if (predicted_class == 0):
            return {"class" : "Negative", "probability": f"{predicted_probability:.2f}%"}
        elif (predicted_class == 1):
            return {"class" : "Positive", "probability": f"{predicted_probability:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Listening to http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug")
