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
        # Prepare data for LSTM
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

        return render(request, 'aqi_prediction/prediction_result.html', {
            'form': form,
            'prediction_datetime': prediction_datetime,
            'pm25_prediction': round(pm25_pred, 2),
            'o3_prediction': round(o3_pred, 2),
            'model_type': model_type
        })

    return render(request, 'aqi_prediction/prediction_form.html', {'form': form})
