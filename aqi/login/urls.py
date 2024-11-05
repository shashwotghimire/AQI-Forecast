from django.urls import path
from . import views

urlpatterns = [
    path('', views.user_login, name='login'),  # Use 'user_login' here
    path('register/', views.register, name='register'),
]
