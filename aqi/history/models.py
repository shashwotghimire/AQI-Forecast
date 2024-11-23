
from django.db import models

class AQIPrediction(models.Model):
    prediction = models.CharField(max_length=50)  # AQI value
    timestamp = models.DateTimeField(auto_now_add=True)  # Automatically sets the time when the prediction is saved

    def __str__(self):
        return f"{self.prediction} ({self.timestamp})"

# Create your models here.
