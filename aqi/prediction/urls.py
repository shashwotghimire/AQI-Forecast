from django.urls import path
from . import views
urlpatterns = [
    path('', views.prediction, name='prediction'),
    path('result/', views.submit_datetime, name='submit_datetime'),
]
