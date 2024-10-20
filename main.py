from fastapi import FastAPI, HTTPException , File , UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional , List
from datetime import datetime
import logging
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from utils import graph_utils


from fastapi.responses import FileResponse
from apscheduler.schedulers.background import BackgroundScheduler
from tasks.data_fetcher import fetch_and_save_data
import os


import pandas as pd


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://glittering-lily-92e49b.netlify.app","http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class GraphRequest(BaseModel):
    graph_type: str
    pollutant: Optional[str] = None
    time_granularity: Optional[str] = 'daily'
    year: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.post("/generate-graph")
async def generate_graph(request: GraphRequest):
    try:
        # Load the appropriate dataset based on the graph type
        if request.graph_type in ['pollutant', 'pollutant_heatmap', 'normalized_monthly_pollutants']:
            df = pd.read_csv("data/Pune_hist_pollution_data_30_sep_24.csv")
            df = graph_utils.preprocess_data(df, request.time_granularity)
        elif request.graph_type in ['correlation_heatmap', 'subindex_correlation_heatmap', 'aqi_bucket_distribution', 'aqi_over_time']:
            df = pd.read_csv("data/dataframeWithSubindexAndAQI.csv")
            basic_df, subindex_df = graph_utils.preprocess_new_data(df)
        else:
            raise HTTPException(status_code=400, detail="Invalid graph type")

        # Generate the appropriate graph based on the request
        if request.graph_type == 'pollutant':
            if not request.pollutant:
                raise HTTPException(status_code=400, detail="Pollutant name is required for this graph type")
            graph_data = graph_utils.plot_pollutant(df, request.pollutant, request.time_granularity)
        elif request.graph_type == 'pollutant_heatmap':
            if not request.pollutant or not request.year:
                raise HTTPException(status_code=400, detail="Pollutant name and year are required for this graph type")
            try:
                graph_data = graph_utils.plot_pollutant_heatmap(df, request.pollutant, request.year)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        elif request.graph_type == 'normalized_monthly_pollutants':
            graph_data = graph_utils.plot_normalized_monthly_pollutants(df, request.time_granularity)
        elif request.graph_type == 'correlation_heatmap':
            pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']
            graph_data = graph_utils.plot_correlation_heatmap(basic_df, pollutants)
        elif request.graph_type == 'subindex_correlation_heatmap':
            pollutants_subindex = ['PM2.5_SubIndex', 'PM10_SubIndex', 'NO2_SubIndex', 'SO2_SubIndex', 'CO_SubIndex', 'O3_SubIndex', 'NH3_SubIndex']
            graph_data = graph_utils.plot_subindex_correlation_heatmap(subindex_df, pollutants_subindex)
        elif request.graph_type == 'aqi_bucket_distribution':
            graph_data = graph_utils.plot_aqi_bucket_distribution(basic_df)
        elif request.graph_type == 'aqi_over_time':
            graph_data = graph_utils.plot_aqi_over_time(basic_df, request.time_granularity)

        return {"graph_data": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the model file
model_path = "models/lstm_aqi_model.pth"
model = None

# Initialize device and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler()

# Define the model structure (same as during training)
class LSTMAQIModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMAQIModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to load the model if not already loaded
def load_model():
    global model
    if model is None:
        if os.path.exists(model_path):
            try:
                model = LSTMAQIModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="Model is not available yet.")

# Endpoint to predict AQI for the next 7 days
@app.post("/predict")
async def predict_next_7_days():
    try:
        # Load the model
        load_model()

        # Use the latest data from your dataframe for prediction (assuming it's already scaled)
        # Replace this with the actual method you use to get the most recent data for prediction
        df = pd.read_csv("data/new_df.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['AQI'] = scaler.fit_transform(df[['AQI']])

        lookback = 48  # same as during training
        last_data = df['AQI'].values[-lookback:]  # Get the last 48 hours of AQI data
        last_data = last_data.reshape(-1, 1)

        # Function to predict future values
        def predict_future(model, X_input, future_steps=168):
            predictions = []
            for _ in range(future_steps):
                X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device).unsqueeze(0)
                with torch.no_grad():
                    pred = model(X_tensor).cpu().item()
                predictions.append(pred)
                X_input = np.append(X_input[1:], pred).reshape(-1, 1)  # Shift the input window
            return predictions

        # Predict the next 168 hours (7 days)
        predictions = predict_future(model, last_data, future_steps=168)
        predicted_aqi = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Plot the prediction
        plt.figure(figsize=(10, 6))
        plt.plot(predicted_aqi, label="Predicted AQI for next 7 days", color='blue')
        plt.xlabel('Hours')
        plt.ylabel('AQI')
        plt.title('AQI Forecast for Next 7 Days')
        plt.legend()

        # Save the plot as an image in base64 format
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()

        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Return the predicted values and the graph image
        return {
            "predictions": predicted_aqi.flatten().tolist(),
            "image_base64": img_base64
        }

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
    

'''@app.post("/upload_model")
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
    
    return {"message": "Model uploaded successfully"}'''


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use 8000 for local, or $PORT for Render
    uvicorn.run(app, host="0.0.0.0", port=port)