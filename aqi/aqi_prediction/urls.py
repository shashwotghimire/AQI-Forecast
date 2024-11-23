from django.urls import path
from . import views

app_name = 'aqi_prediction'

urlpatterns = [
    path('predict/', views.predict_aqi, name='predict'),
]
