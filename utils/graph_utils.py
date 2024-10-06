import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def save_figure():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def generate_boxplot(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']])
    plt.title("Air Quality Parameters Distribution")
    plt.ylabel("Concentration")
    return save_figure()

def generate_heatmap(data):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data[['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title("Correlation Heatmap of Air Quality Parameters")
    return save_figure()

def generate_timeseries(data):
    plt.figure(figsize=(14, 6))
    for column in ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']:
        plt.plot(data['datetime'], data[column], label=column)
    plt.xlabel('Date')
    plt.ylabel('Concentration')
    plt.title('Air Quality Parameters Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    return save_figure()

def generate_scatterplot(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='pm2_5', y='pm10', hue='no2', palette='viridis')
    plt.title('PM2.5 vs PM10 (colored by NO2 levels)')
    plt.xlabel('PM2.5')
    plt.ylabel('PM10')
    return save_figure()

def generate_histogram(data):
    plt.figure(figsize=(12, 6))
    data[['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']].hist(bins=50, figsize=(12, 6))
    plt.suptitle('Histograms of Air Quality Parameters')
    plt.tight_layout()
    return save_figure()

def generate_lineplot(data):
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=data, x='datetime', y='pm2_5', label='PM2.5')
    sns.lineplot(data=data, x='datetime', y='pm10', label='PM10')
    plt.title('PM2.5 and PM10 Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('Concentration')
    plt.legend()
    plt.xticks(rotation=45)
    return save_figure()