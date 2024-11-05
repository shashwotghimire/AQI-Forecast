from django.db import models

class AirQuality(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)  # This automatically sets the timestamp
    pm25 = models.FloatField(default=0)  # Default to 0 if not specified
    o3 = models.FloatField(default=0)  # Default to 0 if not specified

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"AQI Reading at {self.timestamp}"
