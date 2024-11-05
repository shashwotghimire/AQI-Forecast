from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_aqi, name='predict_aqi'),
]
