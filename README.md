Rossmann Store Sales Forecasting: End-to-End ML System

Executive Summary
This project demonstrates the complete lifecycle of a production-ready Machine Learning system designed to forecast 6 weeks of daily sales for Rossmann retail stores. Moving beyond standard notebook-based data science, this project encompasses robust data engineering, pipeline serialization, strict API guardrails, and continuous deployment to the cloud.
The final model achieves a Mean Absolute Percentage Error (MAPE) of ~9.7%, providing highly accurate revenue predictions to assist store managers with:
    • Inventory Planning: Optimizing stock levels based on predicted foot traffic and sales volume.
    • Workforce Allocation: Staffing appropriately for high-demand days (e.g., state holidays or active promotions).
    • Financial Forecasting: Projecting quarterly revenue with high statistical confidence.

System Architecture
Our architecture follows a modern, stateless microservice design:
    1. Data Layer: Raw data ingestion and ETL via Pandas.
    2. Modeling Layer: Scikit-Learn pipelines and XGBoost training (Kaggle Environment).
    3. Serving Layer: FastAPI REST endpoint with Pydantic guardrails.
    4. Hosting & CI/CD: GitHub Webhooks automatically trigger builds on Render (Platform-as-a-Service).

Technology Stack
    • Data Processing & Analysis: pandas, numpy
    • Machine Learning: scikit-learn (Pipelines, ColumnTransformer), xgboost (v3.2.0)
    • API Development: fastapi, uvicorn, pydantic
    • Model Serialization: joblib
    • Version Control & CI/CD: Git, GitHub, Render Auto-Deploy

Phase 1: Data Engineering & Preprocessing
The foundation of the model relies on extracting maximum variance from time-series data while rigorously avoiding future leakage.
    • Temporal Feature Extraction: Deconstructed raw dates into DayOfWeek, Month, Year, and WeekOfYear to allow the tree-based model to capture seasonality.
    • Feature Translation: Engineered binary flags, such as translating complex StateHoliday strings into a strict boolean is_state_holiday feature.
    • Rolling Windows & Lags: Created 7-day rolling means and lag features to capture short-term sales momentum.
    • Leakage Prevention: Strictly dropped the Customers and Sales columns from the training features, ensuring the model only trains on data available before the predicted day occurs.
    
Phase 2: Model Training & Optimization
To ensure the model is robust and immune to bad data in production, preprocessing was baked directly into the model object.
    • Scikit-Learn Pipeline: Implemented a ColumnTransformer to handle missing value imputation (median/mode) and One-Hot Encoding for categorical variables (StoreType, Assortment).
    • Algorithm: Trained an XGBRegressor, chosen for its unparalleled performance on structured tabular data and handling of sparse matrices.
    • Hyperparameter Tuning: Utilized Grid Search to optimize learning_rate, max_depth, and n_estimators, balancing the bias-variance tradeoff to achieve the ~9.7% MAPE.
    
Phase 3: Deployment & API Engineering
The trained pipeline was serialized (.pkl) and wrapped in a stateless REST API, allowing client applications to request predictions securely over the internet.

Live API (Swagger UI): https://xgboost-sales-forecasting.onrender.com/docs

Bulletproof Input Validation (Pydantic Guardrails)
To prevent server crashes from malformed client requests, the API utilizes strict Pydantic schemas. It employs Regex pattern matching and mathematical boundaries to reject bad data before it touches the XGBoost model.

Example Validation Rules:
    • StoreType: Must strictly match regex ^[abcd]$.
    • DayOfWeek: Bounded strictly between 1 and 7.
    • Store: Must be a valid integer between 1 and 1115.
    
Example Valid Payload:
JSON
{
  "Store": 876,
  "DayOfWeek": 1,
  "Open": 1,
  "Promo": 1,
  "StateHoliday": "0",
  "SchoolHoliday": 0,
  "StoreType": "a",
  "Assortment": "c",
  "CompetitionDistance": 250,
  "Promo2": 1
}

Dynamic Feature Handling
Because the serialized model expects historical lag features (which a user cannot reasonably input via a UI), the API includes a dynamic fallback mechanism. It isolates the user's explicit inputs and silently constructs the missing background features (filling them with baseline 0s) to satisfy the XGBoost feature matrix requirements without crashing.


🔄 Phase 4: CI/CD & Environment Management
    • Dependency Pinning: Resolved production environment mismatches by strictly pinning versions (scikit-learn==1.6.1, xgboost==3.2.0) in the requirements.txt, ensuring mathematical parity between the Kaggle training environment and the Render production server.
    • Continuous Deployment: Connected the GitHub repository to Render via webhooks. Any push to the main branch automatically triggers a zero-downtime server rebuild and redeployment.

📌 Future MLOps Enhancements
While this system is production-ready, enterprise scaling would include:
    1. Feature Store Integration: Connecting the API to a live Redis or SQL database to dynamically pull real-time CompetitionDistance and rolling_mean lag features at the time of inference.
    2. Data Drift Monitoring: Implementing tools like Evidently AI to monitor when live store data distributions diverge from the original Kaggle training data.
    3. Automated Retraining: Building a CRON job to retrain the XGBoost model monthly as new sales data becomes available.
