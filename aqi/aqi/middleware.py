from django.shortcuts import redirect
from django.urls import reverse
from django.conf import settings

class RedirectAnonymousUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        if not request.user.is_authenticated and request.path not in [
            reverse('register'),
            reverse('login'),
            # Add other public URLs here
        ]:
            return redirect('register')
        
        response = self.get_response(request)
        return response 