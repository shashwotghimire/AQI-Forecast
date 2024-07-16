from django.shortcuts import render
# from django.http import HttpResponse

def home(request):
    return render(request,'website\\index.html')
    # return HttpResponse('works')          for testing

def about(request):
    return render(request,'website\\about.html')


