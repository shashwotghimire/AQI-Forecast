from django.urls import path

from . import views

urlpatterns = [
    path('', views.prediction, name='prediction'),
    path('prediction_result/', views.model_view, name='submit_datetime'),
]
