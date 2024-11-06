from django.shortcuts import render
from .forms import PredictionForm
from .models import AQIData
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def prepare_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    # Add cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    return df

def create_sequences(X, y, sequence_length=24):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

def train_model(model_type):
    data = AQIData.objects.all().values()
    df = pd.DataFrame(data)
    df = prepare_features(df)

    base_features = ['hour', 'day', 'month', 'day_of_week']
    lstm_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                    'hour', 'day', 'month', 'day_of_week']
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    
    if model_type == 'lstm':
        X = df[lstm_features].values
        y = np.stack([df['pm25'].values, df['o3'].values], axis=1)
        
        # Scale features and targets
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Create sequences for LSTM
        sequence_length = 24  # Use last 24 hours of data
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
        
        # Build improved LSTM model
        model = Sequential([
            LSTM(128, input_shape=(sequence_length, len(lstm_features)), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(2)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return (model, (scaler_X, scaler_y), sequence_length), lstm_features
        
    else:
        X = df[base_features]
        X_scaled = scaler_X.fit_transform(X)
        
        if model_type == 'svm':
            pm25_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            o3_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_type == 'random_forest':
            pm25_model = RandomForestRegressor(n_estimators=100, max_depth=15)
            o3_model = RandomForestRegressor(n_estimators=100, max_depth=15)
        elif model_type == 'knn':
            pm25_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
            o3_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        elif model_type == 'decision_tree':
            pm25_model = DecisionTreeRegressor(max_depth=10, min_samples_split=5)
            o3_model = DecisionTreeRegressor(max_depth=10, min_samples_split=5)
            
        pm25_model.fit(X_scaled, df['pm25'])
        o3_model.fit(X_scaled, df['o3'])
        
        return (pm25_model, o3_model), scaler_X, base_features

def calculate_overall_aqi(pm25, o3):
    aqi_pm25 = pm25 * 1.1
    aqi_o3 = o3 * 1.2
    return max(aqi_pm25, aqi_o3)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk.", "No health implications; enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.", "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected.", "Consider reducing prolonged outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.", "Limit prolonged outdoor exertion, especially sensitive groups."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone.", "Avoid all outdoor activities, if possible."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected.", "Stay indoors and avoid physical activities outside."

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
            result, features = train_model(model_type)
            model, (scaler_X, scaler_y), sequence_length = result
            
            # Get the last sequence_length records from the database
            recent_data = AQIData.objects.all().order_by('-datetime')[:sequence_length]
            recent_df = pd.DataFrame(list(recent_data.values()))
            recent_df = prepare_features(recent_df)
            
            # Prepare sequence for prediction
            X = recent_df[features].values
            X_scaled = scaler_X.transform(X)
            X_seq = X_scaled.reshape(1, sequence_length, len(features))
            
            # Make prediction and inverse transform
            prediction_scaled = model.predict(X_seq)
            prediction = scaler_y.inverse_transform(prediction_scaled)
            pm25_pred, o3_pred = prediction[0]
            
        else:
            models, scaler, features = train_model(model_type)
            pm25_model, o3_model = models
            
            # Prepare features
            feature_data = pred_df[features]
            feature_data_scaled = scaler.transform(feature_data)
            
            # Make predictions
            pm25_pred = pm25_model.predict(feature_data_scaled)[0]
            o3_pred = o3_model.predict(feature_data_scaled)[0]

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