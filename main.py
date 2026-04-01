from fastapi import FastAPI, Request
import joblib
import pandas as pd

# 1. Initialize the API
app = FastAPI(title="Rossmann Sales API")

# 2. Load the model into memory
model = joblib.load('rossmann_xgboost_v2.pkl')

# 3. Create a "Home Page" so we know the server is awake
@app.get("/")
def home():
    return {"message": "Rossmann Sales Forecasting API is Live!"}

# 4. Create the Prediction Endpoint
@app.post("/predict")
async def predict_sales(request: Request):
    try:
        # Get the JSON data sent by the user/app
        data = await request.json()
        
        # Convert it to a DataFrame
        df = pd.DataFrame([data])
        
        # Predict!
        prediction = model.predict(df)[0]
        
        # Send back the result
        return {"predicted_sales_euros": round(float(prediction), 2)}
        
    except Exception as e:
        return {"error": str(e)}
