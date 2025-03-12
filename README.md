# AQI-Forecast

AQI-Forecast is an open-source web application for forecasting the Air Quality Index (AQI) in Kathmandu using Django and TensorFlow. It provides real-time AQI information with forecast upto 1 month ahead for users to monitor air quality.

## Project Architecture

- Django for the backend.
- Tailwind CSS for frontend styling.
- TensorFlow for machine learning models.
- Modular design for scalability.

## Prerequisites

- Python 3.8+ installed.


## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/AQI-Forecast.git
   cd AQI-Forecast
   ```

2. **Create and activate a Python virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install django
   pip install -r requirements.txt
   pip install tensorflow
   ```

4. **Initialize Tailwind CSS:**

   ```bash
   python manage.py tailwind init
   ```

5. **Apply database migrations:**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Load AQI data:**
   Provide the path to your CSV file:
   ```bash
   python manage.py load_aqi_data "path/to/your/csvfile.csv"
   ```

## Usage

1. **Start the Django development server:**

   ```bash
   python manage.py runserver
   ```

2. **Start Tailwind in a separate terminal:**
   Make sure the virtual environment is activated.

   ```bash
   python manage.py tailwind start
   ```

3. **Access the application:**
   Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Contributing

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with detailed descriptions.

For any issues or feature requests, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on the GitHub repository or contact the maintainers.
