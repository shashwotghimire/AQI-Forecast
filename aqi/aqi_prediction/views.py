from django.http import HttpResponse
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from .models import AQIPrediction
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from django.db.models import Avg

# Define features globally at the top of the file
BASE_FEATURES = [
    'pm25_lag1',
    'pm25_lag2',
    'pm25_lag3',
    'pm25_lag24',
    'pm25_rolling_mean_6',
    'pm25_rolling_mean_12',
    'pm25_rolling_mean_24',
    'pm25_trend',
    'month',
    'month_sin',
    'month_cos',
    'is_peak_hour'
]

def prepare_features(df):
    """Optimized feature engineering based on importance"""
    # Core temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Enhanced cyclical encoding for month (high importance)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Enhanced seasonal features (month is important)
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    df['season_sin'] = np.sin(2 * np.pi * df['season']/4)
    df['season_cos'] = np.cos(2 * np.pi * df['season']/4)
    
    # Day-based features (day has high importance)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    
    return df

def create_sequences(X, y, sequence_length=24):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

def print_performance_metrics(y_true, y_pred, model_name, feature_importance=None, features=None):
    """Print comprehensive performance metrics"""
    print("\n" + "="*50)
    print(f"{model_name} Performance Metrics")
    print("="*50)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nRMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Calculate MAPE safely
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        print(f"MAPE: {mape:.2f}%")
    else:
        print("MAPE: N/A (zero values in true data)")
    
    # Error distribution
    errors = y_true - y_pred
    print(f"\nError Distribution:")
    print(f"Mean Error: {np.mean(errors):.2f}")
    print(f"Std Error: {np.std(errors):.2f}")
    print(f"Max Error: {np.max(np.abs(errors)):.2f}")
    
    # Feature importance if available
    if feature_importance is not None and features is not None:
        print("\nFeature Importances:")
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(importance_df)
    
    print("\n" + "="*50)
    return rmse, mae, r2, mape if mask.any() else None

def train_model(model_type):
    # Load and prepare data
    data = AQIData.objects.all().values()
    df = pd.DataFrame(data)
    
    # Sort by datetime to ensure proper sequence
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Simplified feature set (removed problematic features)
    BASE_FEATURES = [
        'pm25_lag1',
        'pm25_lag2',
        'pm25_lag3',
        'pm25_lag24',
        'pm25_rolling_mean_6',
        'pm25_rolling_mean_12',
        'pm25_rolling_mean_24',
        'pm25_trend',
        'month',
        'month_sin',
        'month_cos',
        'is_peak_hour'
    ]
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Cyclical encodings
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Lagged features with safe handling
    for lag in [1, 2, 3, 24]:
        col_name = f'pm25_lag{lag}'
        df[col_name] = df['pm25'].shift(lag)
        df[col_name] = df[col_name].ffill().bfill()
        # Clip extreme values
        df[col_name] = df[col_name].clip(lower=0, upper=500)
    
    # Rolling means with safe handling
    for window in [6, 12, 24]:
        col_name = f'pm25_rolling_mean_{window}'
        df[col_name] = df['pm25'].rolling(window=window, min_periods=1).mean()
        df[col_name] = df[col_name].ffill().bfill()
        # Clip extreme values
        df[col_name] = df[col_name].clip(lower=0, upper=500)
    
    # Trend with safe handling
    df['pm25_trend'] = df['pm25'].diff(periods=3)
    df['pm25_trend'] = df['pm25_trend'].fillna(0)
    # Clip extreme trends
    df['pm25_trend'] = df['pm25_trend'].clip(lower=-100, upper=100)
    
    # Peak hours
    df['is_peak_hour'] = ((df['hour'].isin([7,8,9]) | 
                          df['hour'].isin([17,18,19]))).astype(int)
    
    # Final cleanup using newer methods
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    if model_type == 'random_forest':
        # Split data
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Use the global BASE_FEATURES - don't redefine it here
        
        # Scale features
        scaler_X = StandardScaler()
        
        X_train = train_df[BASE_FEATURES]
        X_test = test_df[BASE_FEATURES]
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Initialize model with modified parameters
        pm25_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=None,
            n_jobs=-1
        )
        
        # Train model
        pm25_model.fit(X_train_scaled, train_df['pm25'])
        
        # Get predictions for both train and test sets
        train_pred = pm25_model.predict(X_train_scaled)
        test_pred = pm25_model.predict(X_test_scaled)
        
        # Print training metrics
        print_performance_metrics(
            train_df['pm25'], train_pred,
            "Random Forest (Training Set)",
            pm25_model.feature_importances_,
            BASE_FEATURES
        )
        
        # Print test metrics
        print_performance_metrics(
            test_df['pm25'], test_pred,
            "Random Forest (Test Set)",
            pm25_model.feature_importances_,
            BASE_FEATURES
        )
        
        return (pm25_model, None), scaler_X, BASE_FEATURES
    
    if model_type == 'lstm':
        # Scale features and targets
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X = df[BASE_FEATURES].values
        y = df[['pm25']].values
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Create sequences for LSTM
        sequence_length = 24  # 24 hours of data
        X_seq, y_seq = [], []
        
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:(i + sequence_length)])
            y_seq.append(y_scaled[i + sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split data
        train_size = int(len(X_seq) * 0.7)
        val_size = int(len(X_seq) * 0.15)
        
        X_train = X_seq[:train_size]
        X_val = X_seq[train_size:train_size+val_size]
        X_test = X_seq[train_size+val_size:]
        
        y_train = y_seq[:train_size]
        y_val = y_seq[train_size:train_size+val_size]
        y_test = y_seq[train_size+val_size:]
        
        # Get input shape
        n_features = len(BASE_FEATURES)
        
        # Build LSTM model with correct input shape
        model = Sequential([
            LSTM(64, input_shape=(sequence_length, n_features), 
                 return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),  # Changed to return_sequences=False
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            
            Dense(1)
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Print model summary for debugging
        print("\nLSTM Model Summary:")
        model.summary()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        train_pred = scaler_y.inverse_transform(train_pred)
        val_pred = scaler_y.inverse_transform(val_pred)
        test_pred = scaler_y.inverse_transform(test_pred)
        
        y_train_orig = scaler_y.inverse_transform(y_train)
        y_val_orig = scaler_y.inverse_transform(y_val)
        y_test_orig = scaler_y.inverse_transform(y_test)
        
        # Print metrics
        print("\nLSTM Model Performance:")
        print("\nTraining Set Metrics:")
        print_performance_metrics(y_train_orig, train_pred, "LSTM (Training)")
        
        print("\nValidation Set Metrics:")
        print_performance_metrics(y_val_orig, val_pred, "LSTM (Validation)")
        
        print("\nTest Set Metrics:")
        print_performance_metrics(y_test_orig, test_pred, "LSTM (Test)")
        
        return (model, (scaler_X, scaler_y), sequence_length), BASE_FEATURES
    
    return None

def compute_sample_weights(y):
    """Compute sample weights based on value distribution"""
    # Calculate weights inversely proportional to value frequency
    weights = np.ones_like(y)
    
    # Give more weight to extreme values
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    weights[y < (q1 - 1.5 * iqr)] = 2.0  # Lower outliers
    weights[y > (q3 + 1.5 * iqr)] = 2.0  # Upper outliers
    
    return weights

def calculate_overall_aqi(pm25, o3):
    # Convert O3 from ppb to ppm if needed
    o3_ppm = o3 / 1000  # Convert from ppb to ppm
    
    # PM2.5 breakpoints (in µg/m³) and corresponding AQI values
    pm25_breakpoints = [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    
    # O3 breakpoints (in ppm) and corresponding AQI values
    o3_breakpoints = [
        (0, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
        (0.201, 0.604, 301, 500),
    ]

    def calculate_aqi(concentration, breakpoints):
        for low_c, high_c, low_aqi, high_aqi in breakpoints:
            if low_c <= concentration <= high_c:
                return round(((high_aqi - low_aqi) / (high_c - low_c) * (concentration - low_c) + low_aqi))
        # If concentration is below the lowest breakpoint, use the lowest AQI
        if concentration < breakpoints[0][0]:
            return breakpoints[0][2]  # Return lowest AQI value
        # If concentration is above the highest breakpoint, use the highest AQI
        return 500  # Above scale

    pm25_aqi = calculate_aqi(pm25, pm25_breakpoints)
    o3_aqi = calculate_aqi(o3_ppm, o3_breakpoints)
    
    return max(pm25_aqi, o3_aqi)

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

def create_prediction_features(records_df, prediction_datetime):
    """Create consistent features for prediction"""
    features = {}
    
    # Basic time features
    features.update({
        'hour': [prediction_datetime.hour],
        'day': [prediction_datetime.day],
        'month': [prediction_datetime.month],
        'day_of_week': [prediction_datetime.weekday()],
    })
    
    # Cyclical features
    features.update({
        'month_sin': [np.sin(2 * np.pi * prediction_datetime.month/12)],
        'month_cos': [np.cos(2 * np.pi * prediction_datetime.month/12)],
        'hour_sin': [np.sin(2 * np.pi * prediction_datetime.hour/24)],
        'hour_cos': [np.cos(2 * np.pi * prediction_datetime.hour/24)]
    })
    
    # Historical features from records_df
    if not records_df.empty:
        latest_record = records_df.iloc[-1]
        features.update({
            'pm25_lag1': [latest_record['pm25']],
            'o3_lag1': [latest_record['o3']],
            'pm25_lag24': [records_df['pm25'].iloc[-24] if len(records_df) >= 24 else latest_record['pm25']],
            'o3_lag24': [records_df['o3'].iloc[-24] if len(records_df) >= 24 else latest_record['o3']],
            'pm25_rolling_mean_24': [records_df['pm25'].tail(24).mean()],
            'o3_rolling_mean_24': [records_df['o3'].tail(24).mean()]
        })
    
    return pd.DataFrame(features)

def evaluate_predictions(y_true, y_pred, target_name):
    """Calculate and return metrics for predictions"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{target_name} Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    
    return rmse, mae, r2

def calculate_correction_factor(recent_records_df, prediction):
    """Calculate correction factor based on recent prediction errors"""
    # Get last few hours of actual values
    recent_actuals = recent_records_df['pm25'].tail(6).values
    
    # Calculate recent trend
    recent_trend = np.mean(np.diff(recent_actuals))
    
    # Calculate average error in recent predictions
    recent_mean = np.mean(recent_actuals)
    prediction_error = recent_mean - prediction
    
    # Combine trend and error for correction
    correction = prediction_error * 0.7 + recent_trend * 0.3
    
    return correction

def get_hourly_variation(hour, base_value):
    """Enhanced hour-specific variation with base value consideration"""
    if 6 <= hour <= 9:  # Morning rush
        return np.random.uniform(2.0, 2.6) * (1 + np.random.normal(0, 0.2))
    elif 10 <= hour <= 15:  # Midday
        return np.random.uniform(1.4, 1.8) * (1 + np.random.normal(0, 0.15))
    elif 16 <= hour <= 20:  # Evening rush
        return np.random.uniform(2.2, 2.8) * (1 + np.random.normal(0, 0.25))
    elif 21 <= hour <= 23:  # Night
        return np.random.uniform(1.2, 1.6) * (1 + np.random.normal(0, 0.15))
    else:  # Late night/early morning
        return np.random.uniform(0.5, 0.7) * (1 + np.random.normal(0, 0.1))

def get_seasonal_variation(month, base_value):
    """Enhanced season-specific variation with base value consideration"""
    if month in [11, 12, 1, 2]:  # Winter
        return np.random.uniform(2.2, 2.8) * (1 + np.random.normal(0, 0.3))
    elif month in [3, 4, 5]:  # Pre-monsoon
        return np.random.uniform(1.8, 2.2) * (1 + np.random.normal(0, 0.25))
    elif month in [6, 7, 8, 9]:  # Monsoon
        return np.random.uniform(0.4, 0.6) * (1 + np.random.normal(0, 0.2))
    else:  # Post-monsoon
        return np.random.uniform(1.4, 1.8) * (1 + np.random.normal(0, 0.25))

def add_dynamic_variation(base_value, hour, month, last_value, recent_values):
    """Enhanced dynamic variation with multiple factors"""
    # Calculate recent statistics
    recent_mean = np.mean(recent_values[-24:]) if len(recent_values) >= 24 else last_value
    recent_std = np.std(recent_values[-24:]) if len(recent_values) >= 24 else last_value * 0.2
    
    # Base variation components
    time_variation = np.random.normal(0, recent_std * 0.3)
    trend_variation = np.random.uniform(-20, 20)
    
    # Time-of-day patterns with randomization
    if 6 <= hour <= 9:  # Morning peak
        time_factor = np.random.uniform(1.2, 1.5)
        extra_variation = base_value * np.random.uniform(0.1, 0.3)
    elif 10 <= hour <= 15:  # Midday
        time_factor = np.random.uniform(0.8, 1.1)
        extra_variation = base_value * np.random.uniform(-0.1, 0.1)
    elif 16 <= hour <= 20:  # Evening peak
        time_factor = np.random.uniform(1.3, 1.6)
        extra_variation = base_value * np.random.uniform(0.2, 0.4)
    elif 21 <= hour <= 23:  # Night
        time_factor = np.random.uniform(0.7, 0.9)
        extra_variation = base_value * np.random.uniform(-0.2, 0)
    else:  # Late night/early morning
        time_factor = np.random.uniform(0.5, 0.7)
        extra_variation = base_value * np.random.uniform(-0.3, -0.1)

    # Seasonal patterns with randomization
    if month in [12, 1, 2]:  # Winter
        season_factor = np.random.uniform(1.4, 1.8)
    elif month in [3, 4, 5]:  # Spring
        season_factor = np.random.uniform(1.2, 1.5)
    elif month in [6, 7, 8]:  # Summer/Monsoon
        season_factor = np.random.uniform(0.6, 0.8)
    else:  # Fall
        season_factor = np.random.uniform(1.0, 1.3)

    # Combine all factors
    adjusted_value = (base_value * time_factor * season_factor + 
                     time_variation + trend_variation + extra_variation)

    # Add random noise based on recent volatility
    noise = np.random.normal(0, recent_std * 0.15)
    adjusted_value += noise

    # Ensure reasonable change from last value
    max_change = last_value * 0.4  # Allow up to 40% change
    if abs(adjusted_value - last_value) > max_change:
        if adjusted_value > last_value:
            adjusted_value = last_value + max_change * np.random.uniform(0.7, 1.0)
        else:
            adjusted_value = last_value - max_change * np.random.uniform(0.7, 1.0)

    return adjusted_value

def predict_aqi(request):
    form = PredictionForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        prediction_datetime = form.cleaned_data['prediction_datetime']
        model_type = form.cleaned_data['model']

        # Get historical records first
        last_records = AQIData.objects.order_by('-datetime')[:24]  # Get at least 24 records
        if len(last_records) < 24:
            return HttpResponse("Need at least 24 historical records")
        
        # Create DataFrame from records
        records_df = pd.DataFrame(list(last_records.values()))
        records_df['datetime'] = pd.to_datetime(records_df['datetime'])
        records_df = records_df.sort_values('datetime')

        # Get recent actual value before model-specific code
        recent_actual = records_df['pm25'].iloc[-1]

        result = train_model(model_type)
        if result is None:
            return HttpResponse("Invalid model type")
        
        if model_type == 'lstm':
            (model, (scaler_X, scaler_y), sequence_length), features = result
            
            # Get exactly sequence_length records
            last_records = AQIData.objects.order_by('-datetime')[:sequence_length]
            if len(last_records) < sequence_length:
                return HttpResponse(f"Need at least {sequence_length} historical records")
            
            records_df = pd.DataFrame(list(last_records.values()))
            records_df['datetime'] = pd.to_datetime(records_df['datetime'])
            records_df = records_df.sort_values('datetime')
            
            # Get recent values for adjustments
            recent_values = records_df['pm25'].values
            recent_o3_values = records_df['o3'].values
            last_value = recent_values[-1]
            last_o3_value = recent_o3_values[-1]
            
            # Time-based factors for Kathmandu
            hour = prediction_datetime.hour
            month = prediction_datetime.month
            
            # Enhanced seasonal adjustments
            winter_factor = 2.2 if month in [11, 12, 1, 2] else 1.0
            pre_monsoon_factor = 1.8 if month in [3, 4, 5] else 1.0
            monsoon_factor = 0.6 if month in [6, 7, 8, 9] else 1.0
            
            # Daily pattern adjustments
            morning_rush = 2.0 if 6 <= hour <= 10 else 1.0
            evening_rush = 2.2 if 16 <= hour <= 20 else 1.0
            night_time = 0.6 if (hour >= 23 or hour <= 4) else 1.0
            
            # O3 specific adjustments
            o3_daytime_factor = 1.8 if 10 <= hour <= 16 else 1.0
            o3_seasonal_factor = 1.5 if month in [3, 4, 5] else 1.0
            
            # Create sequence data
            sequence_data = []
            for _, record in records_df.iterrows():
                feature_dict = {
                    'pm25_lag1': record['pm25'],
                    'pm25_lag2': records_df['pm25'].shift(1).ffill().iloc[-1],
                    'pm25_lag3': records_df['pm25'].shift(2).ffill().iloc[-1],
                    'pm25_lag24': records_df['pm25'].shift(23).ffill().iloc[-1],
                    'pm25_rolling_mean_6': records_df['pm25'].rolling(window=6, min_periods=1).mean().iloc[-1],
                    'pm25_rolling_mean_12': records_df['pm25'].rolling(window=12, min_periods=1).mean().iloc[-1],
                    'pm25_rolling_mean_24': records_df['pm25'].rolling(window=24, min_periods=1).mean().iloc[-1],
                    'pm25_trend': records_df['pm25'].diff(periods=3).fillna(0).iloc[-1],
                    'month': record['datetime'].month,
                    'month_sin': np.sin(2 * np.pi * record['datetime'].month/12),
                    'month_cos': np.cos(2 * np.pi * record['datetime'].month/12),
                    'is_peak_hour': 1 if record['datetime'].hour in [7,8,9,17,18,19] else 0
                }
                sequence_data.append([feature_dict[f] for f in features])
            
            # Prepare sequence for LSTM
            sequence_data = np.array(sequence_data)
            sequence_scaled = scaler_X.transform(sequence_data)
            sequence_scaled = sequence_scaled.reshape(1, sequence_length, len(features))
            
            # Make base prediction
            prediction_scaled = model.predict(sequence_scaled, verbose=0)
            base_pred = scaler_y.inverse_transform(prediction_scaled)[0][0]
            
            # Get recent statistics
            recent_values = records_df['pm25'].values[-24:]
            recent_mean = np.mean(recent_values)
            recent_std = np.std(recent_values)
            
            # Add enhanced dynamic variation
            pm25_pred = add_dynamic_variation(base_pred, hour, month, last_value, recent_values)
            
            # Season-specific bounds with more variation
            if month in [11, 12, 1, 2]:  # Winter
                min_pm25 = max(180 + np.random.uniform(-30, 30), last_value * 0.7)
                max_pm25 = min(900 + np.random.uniform(-50, 50), last_value * 2.0)
            elif month in [3, 4, 5]:  # Pre-monsoon
                min_pm25 = max(120 + np.random.uniform(-20, 20), last_value * 0.7)
                max_pm25 = min(700 + np.random.uniform(-40, 40), last_value * 1.8)
            else:  # Other seasons
                min_pm25 = max(60 + np.random.uniform(-15, 15), last_value * 0.6)
                max_pm25 = min(500 + np.random.uniform(-30, 30), last_value * 1.6)
            
            # Additional time-based adjustments
            if 16 <= hour <= 20:  # Evening peak
                min_pm25 *= np.random.uniform(1.2, 1.4)
                max_pm25 *= np.random.uniform(1.2, 1.4)
                pm25_pred += recent_std * np.random.uniform(0.2, 0.4)
            elif 6 <= hour <= 9:  # Morning peak
                min_pm25 *= np.random.uniform(1.1, 1.3)
                max_pm25 *= np.random.uniform(1.1, 1.3)
                pm25_pred += recent_std * np.random.uniform(0.1, 0.3)
            
            # Final random adjustments
            pm25_pred += np.random.uniform(-10, 10)
            pm25_pred = np.clip(pm25_pred, min_pm25, max_pm25)
            
            # Enhanced O3 prediction
            o3_base = last_o3_value * o3_daytime_factor * o3_seasonal_factor
            o3_pred = add_dynamic_variation(o3_base, hour, month, last_o3_value, recent_o3_values)
            o3_pred = np.clip(o3_pred, 
                             max(25, last_o3_value * 0.6 * np.random.uniform(0.9, 1.1)),
                             min(180, last_o3_value * 1.7 * np.random.uniform(0.9, 1.1)))
            
            print(f"\nEnhanced Prediction Details:")
            print(f"Hour: {hour}, Month: {month}")
            print(f"Last Value: {last_value:.2f}")
            print(f"Base Prediction: {base_pred:.2f}")
            print(f"Final Prediction: {pm25_pred:.2f}")
            print(f"Recent Mean: {recent_mean:.2f}")
            print(f"Recent Std: {recent_std:.2f}")
            print(f"Bounds: [{min_pm25:.2f}, {max_pm25:.2f}]")
            
            # Clip prediction to reasonable range
            pm25_pred = np.clip(pm25_pred, 0, 500)
            
            # Print prediction details
            print(f"\n{model_type.upper()} Prediction Details:")
            print(f"Recent actual PM2.5: {recent_actual:.2f}")
            print(f"Predicted PM2.5: {pm25_pred:.2f}")
            print(f"Prediction delta: {pm25_pred - recent_actual:.2f}")
            print(f"Percent change: {((pm25_pred - recent_actual) / recent_actual * 100):.2f}%")
            
            # Calculate overall AQI
            overall_aqi = calculate_overall_aqi(pm25_pred, recent_actual)  # Using recent_actual for O3
            aqi_category, health_message, health_tip = get_aqi_category(overall_aqi)

            # Save prediction
            new_prediction = AQIPrediction(
                prediction_datetime=prediction_datetime,
                pm25_prediction=pm25_pred,
                o3_prediction=recent_actual,  # Using recent_actual for O3
                overall_aqi=overall_aqi,
                aqi_category=aqi_category,
                model_type=model_type
            )
            new_prediction.save()

            return render(request, 'aqi_prediction/prediction_result.html', {
                'form': form,
                'prediction_datetime': prediction_datetime,
                'pm25_prediction': round(pm25_pred, 2),
                'o3_prediction': round(recent_actual, 2),
                'overall_aqi': round(overall_aqi, 2),
                'model_type': model_type,
                'aqi_category': aqi_category,
                'health_message': health_message,
                'health_tip': health_tip,
                'recent_actual_pm25': round(recent_actual, 2)
            })

        elif model_type == 'random_forest':
            (pm25_model, o3_model), scaler_X, features = result
            
            # Sort and prepare recent data
            records_df = records_df.sort_values('datetime')
            recent_values = records_df['pm25'].values
            recent_o3_values = records_df['o3'].values
            last_value = recent_values[-1]
            last_o3_value = recent_o3_values[-1]
            
            # Time-based factors for Kathmandu
            hour = prediction_datetime.hour
            month = prediction_datetime.month
            
            # Enhanced seasonal adjustments for PM2.5 in Kathmandu
            # Winter months have very high pollution due to temperature inversion
            winter_factor = 2.2 if month in [11, 12, 1, 2] else 1.0  # Increased further
            # Pre-monsoon months have high pollution due to dust
            pre_monsoon_factor = 1.8 if month in [3, 4, 5] else 1.0  # Increased
            # Monsoon months have lower pollution due to rain
            monsoon_factor = 0.6 if month in [6, 7, 8, 9] else 1.0  # Decreased further
            
            # Enhanced daily patterns for PM2.5
            morning_rush = 2.0 if 6 <= hour <= 10 else 1.0  # Increased for morning peak
            evening_rush = 2.2 if 16 <= hour <= 20 else 1.0  # Increased for evening peak
            night_time = 0.6 if (hour >= 23 or hour <= 4) else 1.0  # Decreased for night
            
            # O3 specific adjustments (O3 peaks during midday due to sunlight)
            o3_daytime_factor = 1.8 if 10 <= hour <= 16 else 1.0  # Peak during sunny hours
            o3_seasonal_factor = 1.5 if month in [3, 4, 5] else 1.0  # Higher in pre-monsoon
            
            # Create base features
            input_data = {
                'pm25_lag1': [last_value],
                'pm25_lag2': [records_df['pm25'].iloc[-2] if len(records_df) > 1 else last_value],
                'pm25_lag3': [records_df['pm25'].iloc[-3] if len(records_df) > 2 else last_value],
                'pm25_lag24': [records_df['pm25'].iloc[-24] if len(records_df) >= 24 else last_value],
                'pm25_rolling_mean_6': [records_df['pm25'].tail(6).mean()],
                'pm25_rolling_mean_12': [records_df['pm25'].tail(12).mean()],
                'pm25_rolling_mean_24': [records_df['pm25'].tail(24).mean()],
                'pm25_trend': [records_df['pm25'].diff().tail(3).mean()],
                'month': [month],
                'month_sin': [np.sin(2 * np.pi * month/12)],
                'month_cos': [np.cos(2 * np.pi * month/12)],
                'is_peak_hour': [1 if hour in [7,8,9,17,18,19] else 0]
            }
            
            # Create and scale features
            feature_data = pd.DataFrame(input_data)
            feature_data = feature_data[BASE_FEATURES]
            feature_data_scaled = scaler_X.transform(feature_data)
            
            # Make base predictions
            base_pred = pm25_model.predict(feature_data_scaled)[0]
            
            # Apply Kathmandu-specific adjustments for PM2.5
            seasonal_adjustment = base_pred * (winter_factor * pre_monsoon_factor * monsoon_factor - 1.0)
            daily_adjustment = base_pred * (morning_rush * evening_rush * night_time - 1.0)
            
            # Calculate recent trends and variability
            recent_trend = np.mean(np.diff(recent_values[-6:])) if len(recent_values) >= 6 else 0
            recent_std = np.std(recent_values[-24:]) if len(recent_values) >= 24 else np.std(recent_values)
            
            # Enhanced random adjustment based on time of day
            hour_factor = np.sin(2 * np.pi * hour / 24)
            random_adjustment = np.random.normal(0, recent_std * 0.2) * (1 + abs(hour_factor))
            
            # Combine all adjustments for PM2.5
            pm25_pred = base_pred + seasonal_adjustment + daily_adjustment + random_adjustment
            
            # Add trend component with increased weight
            trend_adjustment = recent_trend * 3  # Increased weight on trend
            pm25_pred += trend_adjustment
            
            # Additional evening boost for PM2.5
            if 16 <= hour <= 20:
                evening_boost = base_pred * 0.4  # Increased evening boost
                pm25_pred += evening_boost
            
            # Adjust O3 prediction based on time of day and season
            o3_pred = last_o3_value * o3_daytime_factor * o3_seasonal_factor
            
            # PM2.5 bounds for Kathmandu (based on historical data)
            if winter_factor > 1:  # Winter
                min_pm25 = max(150, last_value * 0.7)
                max_pm25 = min(800, last_value * 1.8)
            elif pre_monsoon_factor > 1:  # Pre-monsoon
                min_pm25 = max(100, last_value * 0.7)
                max_pm25 = min(600, last_value * 1.6)
            else:  # Other seasons
                min_pm25 = max(50, last_value * 0.6)
                max_pm25 = min(400, last_value * 1.4)
            
            # O3 bounds (ppb)
            min_o3 = max(20, last_o3_value * 0.7)
            max_o3 = min(150, last_o3_value * 1.5)
            
            # Clip predictions to bounds
            pm25_pred = np.clip(pm25_pred, min_pm25, max_pm25)
            o3_pred = np.clip(o3_pred, min_o3, max_o3)
            
            print(f"\nDebug Info:")
            print(f"DateTime: {prediction_datetime}")
            print(f"Last PM2.5: {last_value:.2f}, Predicted PM2.5: {pm25_pred:.2f}")
            print(f"Last O3: {last_o3_value:.2f}, Predicted O3: {o3_pred:.2f}")
            print(f"Seasonal factors - Winter: {winter_factor:.1f}, Pre-monsoon: {pre_monsoon_factor:.1f}, Monsoon: {monsoon_factor:.1f}")
            print(f"Time factors - Morning: {morning_rush:.1f}, Evening: {evening_rush:.1f}, Night: {night_time:.1f}")
            print(f"O3 factors - Daytime: {o3_daytime_factor:.1f}, Seasonal: {o3_seasonal_factor:.1f}")
        
        # Clip prediction to reasonable range
        pm25_pred = np.clip(pm25_pred, 0, 500)
        
        # Print prediction details
        print(f"\n{model_type.upper()} Prediction Details:")
        print(f"Recent actual PM2.5: {recent_actual:.2f}")
        print(f"Predicted PM2.5: {pm25_pred:.2f}")
        print(f"Prediction delta: {pm25_pred - recent_actual:.2f}")
        print(f"Percent change: {((pm25_pred - recent_actual) / recent_actual * 100):.2f}%")
        
        # Calculate overall AQI
        overall_aqi = calculate_overall_aqi(pm25_pred, recent_actual)  # Using recent_actual for O3
        aqi_category, health_message, health_tip = get_aqi_category(overall_aqi)

        # Save prediction
        new_prediction = AQIPrediction(
            prediction_datetime=prediction_datetime,
            pm25_prediction=pm25_pred,
            o3_prediction=recent_actual,  # Using recent_actual for O3
            overall_aqi=overall_aqi,
            aqi_category=aqi_category,
            model_type=model_type
        )
        new_prediction.save()

        return render(request, 'aqi_prediction/prediction_result.html', {
            'form': form,
            'prediction_datetime': prediction_datetime,
            'pm25_prediction': round(pm25_pred, 2),
            'o3_prediction': round(recent_actual, 2),
            'overall_aqi': round(overall_aqi, 2),
            'model_type': model_type,
            'aqi_category': aqi_category,
            'health_message': health_message,
            'health_tip': health_tip,
            'recent_actual_pm25': round(recent_actual, 2)
        })

    return render(request, 'aqi_prediction/prediction_form.html', {'form': form})