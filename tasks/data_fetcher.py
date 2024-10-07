# tasks/data_fetcher.py
import requests
import pandas as pd
import datetime

def get_hist_data(lat, lon, start, end, api_key):
    response = requests.get(
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    )
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to get data, status code: {response.status_code}")
        print(f"Reason: {response.reason}")
        return None

def process_data(data):
    records = []
    for entry in data['list']:
        record = {
            'datetime': datetime.datetime.fromtimestamp(entry['dt']),
            'pm2_5': entry['components']['pm2_5'],
            'pm10': entry['components']['pm10'],
            'no2': entry['components']['no2'],
            'so2': entry['components']['so2'],
            'co': entry['components']['co'],
            'no': entry['components']['no'],
            'o3': entry['components']['o3'],
            'nh3': entry['components']['nh3']
        }
        records.append(record)
    return pd.DataFrame(records)

def fetch_and_save_data():
    lat = 18.5204  # Pune coordinates
    lon = 73.8567
    api_key = "b117f05cbf7133c20380710a8ad5a06d"
    
    # Set the start date to 24 hours before the current date
    #end_date = datetime.datetime.now()
    #start_date = end_date - datetime.timedelta(days=1)
    
    #start = int(start_date.timestamp())
    #end = int(end_date.timestamp())

    start_date = datetime.datetime(2024, 10, 1) #historical pollution data is available from this date
    end_date = datetime.datetime.now()
    start = int(start_date.timestamp())
    end = int(end_date.timestamp())
    
    historical_data = get_hist_data(lat, lon, start, end, api_key)
    
    if historical_data:
        df = process_data(historical_data)
        
        # Append to the CSV file if it exists, otherwise create a new one
        csv_path = "data/Pune_hist_pollution_data_30_sep_24.csv"
        df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)
        print("Data saved successfully!")
    else:
        print("No data fetched!")
