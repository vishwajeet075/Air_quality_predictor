import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import calendar
import numpy as np
import io
import base64

def preprocess_data(df, time_granularity='daily'):
    df.replace(-9999.00000, np.nan, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.interpolate(method='time')

    if time_granularity == 'daily':
        df = df.resample('D').mean()
    return df

def preprocess_new_data(df):
    df.rename(columns={'AQI_calculated': 'AQI', 'AQI_bucket_calculated': 'AQI_bucket'}, inplace=True)
    df.dropna(inplace=True)
    basic_val_columns = ['datetime', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3', 'AQI', 'AQI_bucket']
    subindex_columns = ['datetime', 'PM2.5_SubIndex', 'PM10_SubIndex', 'NO2_SubIndex', 'SO2_SubIndex', 'CO_SubIndex', 'O3_SubIndex', 'NH3_SubIndex', 'AQI', 'AQI_bucket']
    basic_val_df = df[basic_val_columns]
    subindex_df = df[subindex_columns]
    return basic_val_df, subindex_df

def plot_pollutant(df, pollutant, time_granularity='daily'):
    plt.figure(figsize=(15, 6))
    sns.set(style="whitegrid")
    plt.plot(df.index, df[pollutant], label=pollutant)
    plt.title(f'{time_granularity.capitalize()} Average of {pollutant.upper()}')
    plt.xlabel('Date')
    plt.ylabel(f'{pollutant.upper()} Concentration')
    plt.legend()
    plt.xticks(rotation=90)
    if time_granularity == 'daily':
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif time_granularity == 'hourly':
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_pollutant_heatmap(df, pollutant, year):
    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Filter the data for the specified year
    yearly_data = df[df.index.year == year].copy()
    
    # Check if there's data for the specified year and pollutant
    if yearly_data.empty or pollutant not in yearly_data.columns:
        raise ValueError(f"No data available for pollutant {pollutant} in year {year}")
    
    yearly_data['day'] = yearly_data.index.day
    yearly_data['month'] = yearly_data.index.month
    
    # Create pivot table with error handling
    try:
        pivot_table = yearly_data.pivot(index='month', columns='day', values=pollutant)
    except ValueError as e:
        raise ValueError(f"Error creating pivot table: {str(e)}")
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title(f'Daily Average of {pollutant.upper()} for {year} in ug/m3')
    plt.xlabel('Day of the Month')
    plt.ylabel('Month')
    plt.yticks(ticks=[i + 0.5 for i in range(12)], labels=[calendar.month_name[i] for i in range(1, 13)], rotation=0)
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the plot to free up memory
    
    return base64.b64encode(img.getvalue()).decode()

def plot_normalized_monthly_pollutants(df, time_granularity='monthly'):
    plt.figure(figsize=(20, 8))
    if time_granularity == 'monthly':
        monthly_data = df.resample('M').mean()
        normalized_monthly_data = monthly_data / monthly_data.max()
        df = normalized_monthly_data.reset_index()
        plt.title('Monthly Average Concentrations of Pollutants')
        plt.xlabel('Month')
    elif time_granularity == 'yearly':
        yearly_data = df.resample('Y').mean()
        normalized_yearly_data = yearly_data / yearly_data.max()
        df = normalized_yearly_data.reset_index()
        plt.title('Yearly Average Concentrations of Pollutants')
        plt.xlabel('Year')
    plt.ylabel('Concentration')
    monthly_melted = pd.melt(df, id_vars=['datetime'], var_name='Pollutant', value_name='Concentration')
    sns.lineplot(data=monthly_melted, x='datetime', y='Concentration', hue='Pollutant', marker='o')
    plt.xticks(rotation=45)
    plt.legend(title='Pollutant')
    plt.grid(True)
    plt.xticks(pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='M').to_pydatetime(), rotation=90, ha='right')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_correlation_heatmap(basic_val_df, pollutants):
    correlation_df = basic_val_df[pollutants + ['AQI']]
    correlation_matrix = correlation_df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Pollutants and AQI')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_subindex_correlation_heatmap(subindex_df, pollutants_subindex):
    correlation_df = subindex_df[pollutants_subindex + ['AQI']]
    correlation_matrix = correlation_df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Pollutant Subindices and AQI')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_aqi_bucket_distribution(basic_val_df):
    aqi_bucket_counts = basic_val_df.groupby('AQI_bucket').size()
    plt.figure(figsize=(10, 6))
    aqi_bucket_counts.plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    plt.gca().spines[['top', 'right']].set_visible(True)
    plt.title('AQI Bucket Distribution')
    plt.xlabel('Count')
    plt.ylabel('AQI Bucket')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_aqi_over_time(df, time_granularity='hourly'):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.loc[:, 'datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(40, 15))
    if time_granularity == 'daily':
        df_resampled = numeric_df.resample('D').mean()
    elif time_granularity == 'monthly':
        df_resampled = numeric_df.resample('M').mean()
    else:
        df_resampled = numeric_df
    plt.plot(df_resampled.index, df_resampled['AQI'], label='AQI', color='b', linestyle='-', marker='o')
    plt.title(f'{time_granularity.capitalize()} AQI Values Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()