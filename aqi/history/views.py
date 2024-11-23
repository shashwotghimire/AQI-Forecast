from django.shortcuts import render
from aqi_prediction.models import AQIPrediction

def history_view(request):
    predictions = AQIPrediction.objects.all().order_by('-prediction_datetime')
    return render(request, 'history/history.html', {'predictions': predictions})
