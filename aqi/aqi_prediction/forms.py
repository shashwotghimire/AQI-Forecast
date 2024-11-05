from django import forms

class PredictionForm(forms.Form):
    prediction_datetime = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        label='Select Date and Time for Prediction'
    )