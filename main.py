from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
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
    allow_origins=["http://localhost:3000"],
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)