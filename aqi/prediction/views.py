from django.shortcuts import render
from django.views.generic import FormView
from .forms import PredictionForm
from .models import AirQuality  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class PredictionView(FormView):
    template_name = 'prediction/predict.html'
    form_class = PredictionForm
    success_url = '/'

    def form_valid(self, form):
        target_datetime = form.cleaned_data['prediction_datetime']
        predictions = self.predict_aqi(target_datetime)
        return render(self.request, self.template_name, {
            'form': form,
            'predictions': predictions
        })

    def predict_aqi(self, target_datetime):
        # Get all readings
        readings = AirQuality.objects.all().values('timestamp', 'pm25', 'o3')
        df = pd.DataFrame(readings)

        if len(df) == 0:
            return {'pm25': 0, 'o3': 0}

        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Prepare feature matrices
        X = df[['hour', 'day', 'month', 'day_of_week']].values
        y_pm25 = df['pm25'].values
        y_o3 = df['o3'].values

        # Train models
        rf_pm25 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_o3 = RandomForestRegressor(n_estimators=100, random_state=42)

        rf_pm25.fit(X, y_pm25)
        rf_o3.fit(X, y_o3)

        # Prepare prediction features
        pred_features = np.array([[
            target_datetime.hour,
            target_datetime.day,
            target_datetime.month,
            target_datetime.weekday()
        ]])

        # Make predictions
        pm25_pred = rf_pm25.predict(pred_features)[0]
        o3_pred = rf_o3.predict(pred_features)[0]

        return {
            'pm25': round(pm25_pred, 2),
            'o3': round(o3_pred, 2)
        }