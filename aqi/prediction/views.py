from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import pickle
import numpy as np
from datetime import datetime

# Load the model
def load_model():
    with open(settings.MODEL_FILE_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def prediction(request):

    return render(request,'prediction\\prediction.html')

def submit_datetime(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        time = request.POST.get('time')
        
        if date and time:
            date_str = f"{date} {time}:00"  # Combine date and time into the desired format
            print("Combined DateTime:", date_str)  # For debugging purposes, can be removed
            # You can now store date_str in your database or use it as needed
            
            # For demonstration, we return it in the response
            return HttpResponse(f"Combined DateTime: {date_str}")
    
    return render(request, 'prediction\\prediction_result.html')
