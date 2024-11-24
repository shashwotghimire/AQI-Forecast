from django.shortcuts import render
from aqi_prediction.models import AQIPrediction
from django.core.paginator import Paginator
from datetime import datetime

def history_view(request):
    
    selected_date = request.GET.get('date')
    
    # Get base queryset ordered by datetime
    queryset = AQIPrediction.objects.all().order_by('-prediction_datetime')
    
    # Filter by date if selected
    if selected_date:
        try:
            date_obj = datetime.strptime(selected_date, '%Y-%m-%d').date()
            queryset = queryset.filter(prediction_datetime__date=date_obj)
        except ValueError:
            pass  # Invalid date format, show all results
    
    paginator = Paginator(queryset, 25)  
    page = request.GET.get('page')
    predictions = paginator.get_page(page)
    
    return render(request, 'history/history.html', {
        'predictions': predictions,
        'selected_date': selected_date,
    })