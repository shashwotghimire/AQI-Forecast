from django import forms

MODEL_CHOICES = [
    ('lstm', 'LSTM Neural Network'),
    ('svm', 'Support Vector Machine (SVM)'),
    ('random_forest', 'Random Forest'),
    ('knn', 'K-Nearest Neighbors (KNN)'),
    ('decision_tree', 'Decision Tree'),
]

class PredictionForm(forms.Form):
    prediction_datetime = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={
            'type': 'datetime-local',
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'aria-label': 'Select prediction date and time'
        }),
        label='Select Date and Time for Prediction'
    )
    
    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Select Prediction Model',
        initial='random_forest',  
        widget=forms.Select(attrs={
            'class': 'border rounded px-3 py-2 w-full',
            'aria-label': 'Select machine learning model'
        }),
        help_text='Choose the machine learning model for AQI prediction'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model'].widget.attrs.update({
            'onchange': 'this.form.classList.add("changed")'  # Optional: adds visual feedback on model change
        })