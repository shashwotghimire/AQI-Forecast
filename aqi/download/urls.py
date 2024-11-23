from django.urls import path
from . import views

app_name = 'download'

urlpatterns = [
    path('', views.download_page, name='download_page'),  # For the download button
    path('download_csv/', views.download_csv, name='download_csv'),  # For the actual download
]
