from django.shortcuts import render
from .forms import PredictionForm
from .models import AQIData
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def prepare_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df

def train_model(model_type):
    data = AQIData.objects.all().values()
    df = pd.DataFrame(data)
    df = prepare_features(df)

    feature_columns = ['hour', 'day', 'month', 'day_of_week']

    if model_type == 'lstm':
        X = df[feature_columns].to_numpy()
        y = np.stack([df['pm25'].to_numpy(), df['o3'].to_numpy()], axis=1)

        model = Sequential()
        model.add(LSTM(50, input_shape=(X.shape[1], 1)))
        model.add(Dense(2))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X.reshape(-1, X.shape[1], 1), y, epochs=50, batch_size=32)
        return model

    elif model_type == 'svm':
        pm25_model = SVR()
        o3_model = SVR()
        pm25_model.fit(df[feature_columns], df['pm25'])
        o3_model.fit(df[feature_columns], df['o3'])
        return pm25_model, o3_model

    elif model_type == 'random_forest':
        pm25_model = RandomForestRegressor()
        o3_model = RandomForestRegressor()
        pm25_model.fit(df[feature_columns], df['pm25'])
        o3_model.fit(df[feature_columns], df['o3'])
        return pm25_model, o3_model

def calculate_overall_aqi(pm25, o3):
    aqi_pm25 = pm25 * 1.1  # Example factor
    aqi_o3 = o3 * 1.2      # Example factor
    return max(aqi_pm25, aqi_o3)  # Use the higher value as the overall AQI

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is considered satisfactory.", "No health implications; enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable.", "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects.", "Consider reducing prolonged outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects.", "Limit prolonged outdoor exertion, especially sensitive groups."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: everyone may experience more serious health effects.", "Avoid all outdoor activities, if possible."
    else:
        return "Hazardous", "Health warnings of emergency conditions.", "Stay indoors and avoid physical activities outside."

def predict_aqi(request):
    form = PredictionForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        prediction_datetime = form.cleaned_data['prediction_datetime']
        model_type = form.cleaned_data['model']

        pred_df = pd.DataFrame({
            'datetime': [prediction_datetime]
        })
        pred_df = prepare_features(pred_df)

        if model_type == 'lstm':
            model = train_model(model_type)
            X = pred_df[['hour', 'day', 'month', 'day_of_week']].to_numpy().reshape(1, 1, -1)
            pm25_pred, o3_pred = model.predict(X)[0]
        else:
            pm25_model, o3_model = train_model(model_type)
            pm25_pred = pm25_model.predict(pred_df[['hour', 'day', 'month', 'day_of_week']])[0]
            o3_pred = o3_model.predict(pred_df[['hour', 'day', 'month', 'day_of_week']])[0]

        overall_aqi = calculate_overall_aqi(pm25_pred, o3_pred)
        aqi_category, health_message, health_tip = get_aqi_category(overall_aqi)

        return render(request, 'aqi_prediction/prediction_result.html', {
            'form': form,
            'prediction_datetime': prediction_datetime,
            'pm25_prediction': round(pm25_pred, 2),
            'o3_prediction': round(o3_pred, 2),
            'overall_aqi': round(overall_aqi, 2),
            'model_type': model_type,
            'aqi_category': aqi_category,
            'health_message': health_message,
            'health_tip': health_tip
        })

    return render(request, 'aqi_prediction/prediction_form.html', {'form': form})
