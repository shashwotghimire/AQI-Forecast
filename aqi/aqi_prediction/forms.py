from django import forms

MODEL_CHOICES = [
    ('lstm', 'LSTM'),
    ('svm', 'SVM'),
    ('random_forest', 'Random Forest'),
]

class PredictionForm(forms.Form):
    prediction_datetime = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={
            'type': 'datetime-local',
            'class': 'border rounded px-3 py-2 w-full mb-4'  # Added spacing class
        }),
        label='Select Date and Time for Prediction'
    )
    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Select Prediction Model',
        initial='svm',
        widget=forms.Select(attrs={
            'class': 'border rounded px-3 py-2 w-full'  # Added spacing class
        })
    )
