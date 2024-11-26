from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.views.decorators.csrf import csrf_protect
from .forms import CustomUserCreationForm, CustomAuthenticationForm

# Registration View
@csrf_protect
def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Automatically log in the user after registration
            return redirect('home')  # make sure you have a 'home' URL pattern defined
    else:
        form = CustomUserCreationForm()
    return render(request, 'login/register.html', {'form': form})

# Login View
@csrf_protect
def user_login(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  # make sure you have a 'home' URL pattern defined
    else:
        form = CustomAuthenticationForm()
    return render(request, 'login/login.html', {'form': form})
