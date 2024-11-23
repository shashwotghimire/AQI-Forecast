from django.shortcuts import render
import requests
from django.conf import settings

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# from django.http import HttpResponse

# def home(request):
#     return render(request,'website\\index.html')
#     # return HttpResponse('works')          for testing

def about(request):
    return render(request,'website/about.html')

# weather data fetch
def get_weather_data():
    city = "Kathmandu"
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def home(request):
    weather_data = get_weather_data()
    
    context = {
        'weather': weather_data,
    }
    return render(request, 'website/index.html', context)

# aqi data fetch

def get_aqi_data():
    # Latitude and longitude for Kathmandu
    lat = 27.7167
    lon = 85.3167
    api_key = settings.WEATHER_API_KEY  # Use your OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
@login_required
def home(request):
    weather_data = get_weather_data()
    aqi_data = get_aqi_data()
    
    context = {
        'weather': weather_data,
        'aqi': aqi_data,
    }
    return render(request, 'website/index.html', context)



