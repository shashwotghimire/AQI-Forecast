from django.db import models

class AQIData(models.Model):
    datetime = models.DateTimeField()
    pm25 = models.FloatField()
    o3 = models.FloatField()
    
    class Meta:
        ordering = ['datetime']
      

class AQIPrediction(models.Model):
    prediction_datetime = models.DateTimeField()
    pm25_prediction = models.FloatField()
    o3_prediction = models.FloatField()
    overall_aqi = models.FloatField()
    aqi_category = models.CharField(max_length=50)
    model_type = models.CharField(max_length=50, default='default')

    def __str__(self):
        return f"Prediction at {self.prediction_datetime}"
