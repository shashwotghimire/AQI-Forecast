from django.shortcuts import render
from .forms import PredictionForm
from .models import AQIData
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import numpy as np

def prepare_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df

def train_model():
    # Get all data from database
    data = AQIData.objects.all().values()
    df = pd.DataFrame(data)
    
    # Prepare features
    df = prepare_features(df)
    
    # Features for training
    feature_columns = ['hour', 'day', 'month', 'day_of_week']
    
    # Train separate models for PM2.5 and O3
    pm25_model = RandomForestRegressor(n_estimators=100, random_state=42)
    o3_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    pm25_model.fit(df[feature_columns], df['pm25'])
    o3_model.fit(df[feature_columns], df['o3'])
    
    return pm25_model, o3_model

def predict_aqi(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            prediction_datetime = form.cleaned_data['prediction_datetime']
            
            # Create features for prediction
            pred_df = pd.DataFrame({
                'datetime': [prediction_datetime]
            })
            pred_df = prepare_features(pred_df)
            
            # Train models and make predictions
            pm25_model, o3_model = train_model()
            
            feature_columns = ['hour', 'day', 'month', 'day_of_week']
            pm25_pred = pm25_model.predict(pred_df[feature_columns])[0]
            o3_pred = o3_model.predict(pred_df[feature_columns])[0]
            
            return render(request, 'aqi_prediction\\prediction_result.html', {
                'form': form,
                'prediction_datetime': prediction_datetime,
                'pm25_prediction': round(pm25_pred, 2),
                'o3_prediction': round(o3_pred, 2)
            })
    else:
        form = PredictionForm()
    
    return render(request, 'aqi_prediction\\prediction_form.html', {'form': form})
