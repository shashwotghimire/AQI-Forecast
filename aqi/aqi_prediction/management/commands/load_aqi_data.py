# aqi_prediction/management/commands/load_aqi_data.py
from django.core.management.base import BaseCommand
import pandas as pd
from aqi_prediction.models import AQIData
from datetime import datetime
from django.utils import timezone
import pytz

class Command(BaseCommand):
    help = 'Load Kathmandu AQI data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **options):
        try:
            self.stdout.write(f"Loading Kathmandu AQI data from: {options['csv_file']}")
            
            # Read the CSV file
            df = pd.read_csv(options['csv_file'])
            
            # Print data summary
            self.stdout.write(f"Columns found in CSV: {', '.join(df.columns)}")
            self.stdout.write(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
            self.stdout.write(f"Number of records: {len(df)}")
            
            # Basic data statistics
            self.stdout.write("\nData Statistics:")
            self.stdout.write(f"PM2.5 range: {df['PM25'].min():.2f} to {df['PM25'].max():.2f} μg/m³")
            self.stdout.write(f"O3 range: {df['O3'].min():.2f} to {df['O3'].max():.2f} ppb")
            
            # Convert DATE column to timezone-aware datetime
            nepal_tz = pytz.timezone('Asia/Kathmandu')
            df['datetime'] = pd.to_datetime(df['DATE']).apply(
                lambda x: nepal_tz.localize(x) if x is not pd.NaT else None
            )
            
            # Clear existing data
            AQIData.objects.all().delete()
            self.stdout.write("Cleared existing data from database")
            
            # Updated thresholds based on actual Kathmandu data
            PM25_MAX = 600  # μg/m³ (increased to accommodate extreme pollution events)
            O3_MAX = 100    # ppb (adjusted based on observed maximum)
            
            success_count = 0
            error_count = 0
            
            # Add seasonal classification
            def get_season(date):
                month = date.month
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Pre-monsoon'
                elif month in [6, 7, 8, 9]:
                    return 'Monsoon'
                else:
                    return 'Post-monsoon'
            
            # Calculate seasonal statistics
            df['season'] = df['datetime'].apply(get_season)
            seasonal_stats = df.groupby('season').agg({
                'PM25': ['mean', 'max'],
                'O3': ['mean', 'max']
            })
            
            self.stdout.write("\nSeasonal Analysis:")
            for season in seasonal_stats.index:
                self.stdout.write(f"\n{season}:")
                self.stdout.write(f"  Average PM2.5: {seasonal_stats.loc[season, ('PM25', 'mean')]:.2f} μg/m³")
                self.stdout.write(f"  Maximum PM2.5: {seasonal_stats.loc[season, ('PM25', 'max')]:.2f} μg/m³")
                self.stdout.write(f"  Average O3: {seasonal_stats.loc[season, ('O3', 'mean')]:.2f} ppb")
                self.stdout.write(f"  Maximum O3: {seasonal_stats.loc[season, ('O3', 'max')]:.2f} ppb")
            
            for index, row in df.iterrows():
                try:
                    # Validate data
                    if pd.isna(row['datetime']):
                        raise ValueError("Missing datetime")
                    
                    pm25 = float(row['PM25'])
                    o3 = float(row['O3'])
                    
                    # Validate ranges with warnings for extreme values
                    if pm25 > PM25_MAX:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Warning: Extremely high PM2.5 value ({pm25} μg/m³) on {row['datetime']}"
                            )
                        )
                    if o3 > O3_MAX:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Warning: Extremely high O3 value ({o3} ppb) on {row['datetime']}"
                            )
                        )
                    
                    AQIData.objects.create(
                        datetime=row['datetime'],
                        pm25=pm25,
                        o3=o3
                    )
                    success_count += 1
                    
                    if success_count % 1000 == 0:
                        self.stdout.write(f"Processed {success_count} records...")
                        
                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f"Error on row {index}: {str(e)}"
                        )
                    )
                    continue
            
            # Final report
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nData loading completed for Kathmandu AQI data:\n'
                    f'Successfully loaded: {success_count} records\n'
                    f'Failed records: {error_count}\n'
                    f'Location: Kathmandu, Nepal\n'
                    f'Timezone: Asia/Kathmandu (NPT)\n'
                    f'Parameters: PM2.5 (μg/m³), O3 (ppb)\n'
                    f'Date Range: {df["datetime"].min().strftime("%Y-%m-%d")} to {df["datetime"].max().strftime("%Y-%m-%d")}'
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error loading data: {str(e)}')
            )