from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


    from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)



from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)

from django.shortcuts import render
from datetime import datetime, timedelta
import random

class WeatherStats:
    """Utility class for weather statistics and calculations"""
    
    def __init__(self):
        self.temperature_readings = []
        self.humidity_readings = []
        self.wind_speed_readings = []
    
    def add_reading(self, temp, humidity, wind_speed):
        """Add a new weather reading"""
        self.temperature_readings.append(temp)
        self.humidity_readings.append(humidity)
        self.wind_speed_readings.append(wind_speed)
    
    def get_average_temperature(self):
        """Calculate average temperature from readings"""
        if not self.temperature_readings:
            return 0
        return sum(self.temperature_readings) / len(self.temperature_readings)
    
    def get_average_humidity(self):
        """Calculate average humidity from readings"""
        if not self.humidity_readings:
            return 0
        return sum(self.humidity_readings) / len(self.humidity_readings)
    
    def get_wind_speed_range(self):
        """Get min and max wind speeds"""
        if not self.wind_speed_readings:
            return (0, 0)
        return (min(self.wind_speed_readings), max(self.wind_speed_readings))
    
    def generate_mock_data(self, days=7):
        """Generate mock weather data for testing"""
        for _ in range(days * 24):  # Hourly readings
            self.add_reading(
                temp=random.uniform(15, 35),
                humidity=random.uniform(30, 90),
                wind_speed=random.uniform(0, 25)
            )
            
    def clear_data(self):
        """Clear all stored readings"""
        self.temperature_readings.clear()
        self.humidity_readings.clear()
        self.wind_speed_readings.clear()

def index(request):
    """Main weather view"""
    stats = WeatherStats()
    stats.generate_mock_data()
    
    context = {
        'avg_temp': round(stats.get_average_temperature(), 1),
        'avg_humidity': round(stats.get_average_humidity(), 1),
        'wind_range': stats.get_wind_speed_range(),
        'timestamp': datetime.now(),
    }
    
    return render(request, 'weather/index.html', context)


    