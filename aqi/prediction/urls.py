from django.urls import path
from . import views
urlpatterns = [
    path('', views.prediction, name='prediction'),
    path('model/',views.choose_model,name="choose_model")
]
