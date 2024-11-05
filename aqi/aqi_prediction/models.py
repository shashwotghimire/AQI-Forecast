from django.db import models

class AQIData(models.Model):
    datetime = models.DateTimeField()
    pm25 = models.FloatField()
    o3 = models.FloatField()
    
    class Meta:
        ordering = ['datetime']