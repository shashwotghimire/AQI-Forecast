from django.db import models
from django.utils import timezone

# Create your models here.

class AirQuality(models.Model):
  CHAI_TYPE_CHOICES = [
    ('G', 'GOOD'),
    ('S', 'SATISFACTORY'),
    ('MO', 'MODERATE'),
    ('P', 'POOR'),
    ('VP', 'VERYPOOR'),
    ('SE', 'SEVERE'),
  ]

  name = models.CharField(max_length=100)
  image = models.ImageField(upload_to='airquality/')
  date_added = models.DateTimeField(default=timezone.now)
  type = models.CharField(max_length=2, choices=CHAI_TYPE_CHOICES, default='MO')

  def __str__(self):
    return self.name