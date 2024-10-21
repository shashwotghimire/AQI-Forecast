from django.shortcuts import render
from django.http import HttpResponse, Http404
import os
from django.conf import settings
from django.utils.encoding import smart_str

def download_page(request):
     return render(request, 'download\\download.html')

def download_csv(request):
    file_path = os.path.join(settings.MEDIA_ROOT, 'filtered_data.csv')
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=%s' % smart_str('filtered_data.csv')
            return response
    else:
        raise Http404("File not found")
