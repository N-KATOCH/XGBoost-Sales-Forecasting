from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# 1. Initialize API and load the Brain
app = FastAPI(
    title="Rossmann Sales API", 
    description="An enterprise-grade XGBoost API with strict data validation."
)
model = joblib.load('rossmann_xgboost_v2.pkl')

# ==========================================
# THE GUARDRAILS (Pydantic Schema)
# ==========================================
class StoreData(BaseModel):
    # gt=0 means "greater than 0". le=1115 means "less than or equal to 1115".
    Store: int = Field(..., gt=0, le=1115, example=876, description="Unique Id for each store (1-1115)")
    DayOfWeek: int = Field(..., ge=1, le=7, example=1, description="1=Monday, 7=Sunday")
    Open: int = Field(..., ge=0, le=1, example=1, description="0 = closed, 1 = open")
    Promo: int = Field(..., ge=0, le=1, example=1, description="0 = No Promo, 1 = Promo Active")
    
    # pattern="^[0abc]$" is Regex. It strictly forces the user to ONLY input 0, a, b, or c.
    StateHoliday: str = Field(..., pattern="^[0abc]$", example="0", description="0 = None, a = public, b = Easter, c = Christmas")
    SchoolHoliday: int = Field(..., ge=0, le=1, example=0, description="0 = No, 1 = Yes")
    StoreType: str = Field(..., pattern="^[abcd]$", example="a", description="Store models: a, b, c, d")
    Assortment: str = Field(..., pattern="^[abc]$", example="c", description="Assortment: a = basic, b = extra, c = extended")
    
    CompetitionDistance: float = Field(..., ge=0, example=250.0, description="Distance in meters to nearest competitor")
    Promo2: int = Field(..., ge=0, le=1, example=1, description="0 = no continuing promo, 1 = participating")

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def home():
    return {"message": "Rossmann Sales Forecasting API is Live!"}

@app.post("/predict")
def predict_sales(data: StoreData):
    try:
        # 1. Convert the validated Pydantic data into a Pandas DataFrame
        df = pd.DataFrame([data.dict()])
        
        # 2. MINI FEATURE ENGINEERING
        # If your Kaggle pipeline expected the engineered 'is_state_holiday' instead of the raw letters:
        expected_cols = model.feature_names_in_
        if 'is_state_holiday' in expected_cols:
            df['is_state_holiday'] = df['StateHoliday'].apply(lambda x: 0 if x == '0' else 1)
            
        # 3. THE SAFETY NET
        # Fill any complex background columns (like lag features or dates) with 0 so the model doesn't crash
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        # 4. Reorder columns to exactly match the XGBoost training data
        df = df[expected_cols]
        
        # 5. Predict!
        prediction = model.predict(df)[0]
        
        return {
            "status": "success",
            "predicted_sales_euros": round(float(prediction), 2)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
