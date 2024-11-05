# prediction/management/commands/import_aqi_data.py
from django.core.management.base import BaseCommand
from prediction.models import AirQuality
import pandas as pd
from datetime import datetime

class Command(BaseCommand):
    help = 'Import AQI data from CSV file'

    def handle(self, *args, **options):
        try:
            # Use the specific file path
            csv_file_path = r"C:\Users\Dell\Desktop\aqi data.csv"
            
            self.stdout.write(f'Reading file from: {csv_file_path}')
            df = pd.read_csv(csv_file_path)
            
            # Clear existing data (optional)
            AirQuality.objects.all().delete()
            
            # Import new data
            records_created = 0
            for _, row in df.iterrows():
                air_quality = AirQuality(
                    timestamp=datetime.strptime(row['DATE'], '%m/%d/%Y %H:%M'),
                    pm25=row['PM25'],
                    o3=row['O3']
                )
                air_quality.save()
                records_created += 1
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully imported {records_created} AQI readings')
            )
            
        except FileNotFoundError:
            self.stdout.write(
                self.style.ERROR(f'Error: Could not find the file at {csv_file_path}')
            )
        except ValueError as e:
            self.stdout.write(
                self.style.ERROR(f'Error parsing datetime: {str(e)}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error importing data: {str(e)}')
            )