from django.shortcuts import render

# Create your views here.
def prediction(request):
    return render(request,'prediction\prediction.html')

def choose_model(request):
    return render(request,'prediction\models.html')
