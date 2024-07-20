from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import pickle
import numpy as np
from datetime import datetime
from .models import AirQuality

# Load the model
def load_model():
    with open(settings.MODEL_FILE_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def prediction(request):
    return render(request, 'prediction\\prediction.html')

def submit_datetime(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        time = request.POST.get('time')
        
        if date and time:
            date_str = f"{date} {time}:00"  # Combine date and time into the desired format
            print("Combined DateTime:", date_str)  # For debugging purposes
            
            # Example: Load the model
            model = load_model()
            
            # Example: Preprocess the input data (adjust as per your model requirements)
            try:
                input_data = np.array([[datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')]])
                # Example: Make the prediction (replace this with your actual prediction logic)
                prediction_result = model.predict(input_data)
                print("Prediction Result:", prediction_result)  # For debugging purposes
                
                # Pass the prediction result to the new template
                return render(request, 'prediction_result.html', {'prediction': prediction_result, 'date_str': date_str})
            except Exception as e:
                print("Error in prediction:", e)  # For debugging purposes
                return HttpResponse("Error in prediction: {}".format(e))
    
    return HttpResponse("Invalid request method.")

def model_view(request):
    aqi=AirQuality.objects.all()
    return render(request,'prediction\\prediction_result.html',{'aqi':aqi})
