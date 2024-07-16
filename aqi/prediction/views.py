from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponseBadRequest
import numpy as np
import pickle

# Lazy loading for the model
model = None

def load_model():
    global model
    if model is None:
        with open(settings.MODEL_FILE_PATH, 'rb') as file:
            model = pickle.load(file)
    return model

def prediction(request):
    return render(request, 'prediction/prediction.html')

def choose_model(request):
    try:
        # Retrieve input data from the form
        date = request.GET.get('date')
        temperature = float(request.GET.get('temperature'))
        humidity = float(request.GET.get('humidity'))
        precipitation = float(request.GET.get('precipitation'))
        pm25 = float(request.GET.get('pm25'))
        o3 = float(request.GET.get('o3'))

        # Prepare the data for prediction
        input_data = np.array([[temperature, humidity, precipitation, pm25, o3]])

        # Load the model and make prediction
        model = load_model()
        y_pred = model.predict(input_data)

        # Convert numerical prediction back to category
        category_map = {
            0: 'Good',
            1: 'Moderate',
            2: 'Unhealthy for Sensitive Groups',
            3: 'Unhealthy',
            4: 'Very Unhealthy',
            5: 'Hazardous'
        }
        prediction_category = category_map.get(int(y_pred[0]), 'Unknown')

        # Prepare context for the template
        context = {
            'prediction': prediction_category,
            'date': date,
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': precipitation,
            'pm25': pm25,
            'o3': o3
        }
        return render(request, 'prediction/models.html', context)

    except ValueError as e:
        return HttpResponseBadRequest(f"Invalid input: {str(e)}")
    except Exception as e:
        return HttpResponseBadRequest(f"An error occurred: {str(e)}")