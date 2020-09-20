from django import forms
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import *

class createUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class PatientForm(ModelForm):
	class Meta:
		model = Patient
		fields = '__all__'
		#exclude = ['user']

class maladieForm(ModelForm):
    class Meta:
        model = Maladie
        fields = ['maladie_name']

class MessageForm(forms.ModelForm):
    class Meta:
        model = Message
        fields = ['title', 'body', 'pic']
       
class TestForm(forms.ModelForm):
    class Meta:
        model = Test
        fields = ['test_image'] 
   
