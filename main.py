from fastapi import FastAPI, HTTPException , File , UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional , List
from datetime import datetime
import joblib


from fastapi.responses import FileResponse
from apscheduler.schedulers.background import BackgroundScheduler
from tasks.data_fetcher import fetch_and_save_data
import os


import pandas as pd
from utils.graph_utils import (
    generate_boxplot,
    generate_heatmap,
    generate_timeseries,
    generate_scatterplot,
    generate_histogram,
    generate_lineplot
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://glittering-lily-92e49b.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the dataset
data = pd.read_csv('data/Pune_hist_pollution_data_30_sep_24.csv')
data['datetime'] = pd.to_datetime(data['datetime'])

class GraphRequest(BaseModel):
    graph_type: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None

@app.post("/generate_graph")
async def generate_graph(request: GraphRequest):
    graph_type = request.graph_type.lower()
    from_date = pd.to_datetime(request.from_date) if request.from_date else None
    to_date = pd.to_datetime(request.to_date) if request.to_date else None

    # Filter data by date
    filtered_data = data
    if from_date and to_date:
        filtered_data = data[(data['datetime'] >= from_date) & (data['datetime'] <= to_date)]

    if graph_type == "boxplot":
        image_base64 = generate_boxplot(filtered_data)
    elif graph_type == "heatmap":
        image_base64 = generate_heatmap(filtered_data)
    elif graph_type == "timeseries":
        image_base64 = generate_timeseries(filtered_data)
    elif graph_type == "scatterplot":
        image_base64 = generate_scatterplot(filtered_data)
    elif graph_type == "histogram":
        image_base64 = generate_histogram(filtered_data)
    elif graph_type == "lineplot":
        image_base64 = generate_lineplot(filtered_data)
    else:
        raise HTTPException(status_code=400, detail="Invalid graph type")

    return {"image": image_base64}



# Load the model
model_path = "models/model.joblib"
model = joblib.load(model_path)

class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    pm10: float
    no2: float
    so2: float
    co: float
    no: float
    o3: float
    nh3: float

class PredictionResponse(BaseModel):
    pm2_5: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return PredictionResponse(pm2_5=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
# Load the model
model_path = "models/model.joblib"
model = joblib.load(model_path)

class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    pm10: float
    no2: float
    so2: float
    co: float
    no: float
    o3: float
    nh3: float

class PredictionResponse(BaseModel):
    pm2_5: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return PredictionResponse(pm2_5=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Define the model and the path to the model file
model_path = "models/model.joblib"
model = None

class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    pm10: float
    no2: float
    so2: float
    co: float
    no: float
    o3: float
    nh3: float

class PredictionResponse(BaseModel):
    pm2_5: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model

    # Load the model dynamically if not already loaded
    if model is None:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print("Model loaded successfully.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        else:
            # Return a 503 error if the model is not yet available
            raise HTTPException(status_code=503, detail="Model is not available yet. Please try again later.")
    
    try:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([request.dict()])
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        return PredictionResponse(pm2_5=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

'''
# Initialize the scheduler
scheduler = BackgroundScheduler()

# Schedule the task to run every day at noon
#scheduler.add_job(fetch_and_save_data, 'cron', hour=12)
#scheduler.start()
scheduler.add_job(fetch_and_save_data, 'interval', minutes=2)
scheduler.start()


@app.on_event("startup")
def startup_event():
    # Start the scheduler when the app starts
    if not scheduler.running:
        scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    # Shut down the scheduler when the app stops
    scheduler.shutdown()
'''
@app.get("/")
async def root():
    return {"message": "Welcome to the Air Quality Data API"}

@app.get("/data")
async def get_data():
    csv_path = "data/Pune_hist_pollution_data_30_sep_24.csv"
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type='text/csv', filename="data/Pune_hist_pollution_data_30_sep_24.csv")
    else:
        return {"error": "CSV file not found"}
    

@app.post("/upload_model")
async def upload_model(model: UploadFile = File(...)):
    model_path = "models/model.joblib"
    
    # Check if the uploaded file has the correct filename
    if model.filename != "trained_model.joblib":
        return JSONResponse(status_code=422, content={"message": "Invalid file name. Please upload 'trained_model.joblib'."})

    # Save the uploaded model file
    try:
        with open(model_path, "wb") as model_file:
            content = await model.read()  # Read the content of the uploaded file
            model_file.write(content)  # Write content to the specified path
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error saving model: {str(e)}"})
    
    return {"message": "Model uploaded successfully"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use 8000 for local, or $PORT for Render
    uvicorn.run(app, host="0.0.0.0", port=port)