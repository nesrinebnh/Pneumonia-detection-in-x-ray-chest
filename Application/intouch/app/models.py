from django.db import models
from django.contrib.auth.models import User

import numpy as np


from django.utils import timezone

# Create your models here.



class Patient(models.Model):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    prenom = models.CharField(max_length=100)
    age = models.CharField(max_length=3)
    pic = models.ImageField(null=True)
    

class Test(models.Model):
    test_image = models.ImageField() 
    probability = models.CharField(null=True, max_length=100) 
    patient = models.ForeignKey(Patient, null=True, on_delete= models.SET_NULL)
    date = models.DateTimeField(default=timezone.now)  

    def predict(self):
    
        
        
            
        print("ok")
      


       

    
class Maladie(models.Model):
    gender = (
        (None, 'Liste de maladie chronique'),
        ('Diabete', 'Diabete'),
        ('Hypertension arterielle', 'Hypertension arterielle'),
        ('Maladies rhumatismale', 'Maladies rhumatismale'),
        ('Maladies cardiaque', 'Maladies cardiaque'),
        )
    maladie_name = models.CharField(max_length=200, blank=True, choices=gender, verbose_name="gender", unique=True)
    patient = models.ForeignKey(Patient, null=True, on_delete= models.SET_NULL)
    
class Doctor(models.Model):
    specialite = (
        (None, 'Votre specialité'),
        ('Ophtalmologie', 'Ophtalmologie'),
        ('Radiologie', 'Radiologie'),
        ('ORL', 'ORL'),
        ('Cardiologie', 'Cardiologie'),
        ('Pneumologie', 'Pneumologie'),
        ('Neurologie', 'Neurologie'),
        ('Rhumatologie', 'Rhumatologie'),
        ('Infectiologie', 'Infectiologie'),
        ('Hématologie', 'Hématologie'),
        ('Gasto-entéro-hépatologie', 'Gasto-entéro-hépatologie'),
        ('Dermatologie', 'Dermatologie'),
    )
    grade = (
        (None, 'Votre grade'),
        ('Résident', 'Résident'),
        ('Assistant', 'Assistant'),
        ('Maitre assistant', 'Maitre assistant'),
        ('Maitre de conference A', 'Maitre de conference A'),
        ('Maitre de conference B', 'Maitre de conference B'),
        ('Professeur', 'Professeur'),
    )
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.CASCADE)
    nom = models.CharField(max_length=200)
    prenom = models.CharField(max_length=200)
    tlf = models.CharField(max_length=200, null=True)
    grade = models.CharField(max_length=200,null=True ,blank=True, choices=grade, verbose_name="grade")
    specialite = models.CharField(max_length=200, null=True,blank=True, choices=specialite, verbose_name="specialite")
    patient = models.ManyToManyField(Patient)
    pic = models.ImageField(null=True)

class Message(models.Model):
    sender = models.CharField(max_length=200)
    receiver = models.CharField(max_length=200)
    title = models.CharField(max_length=300, null=True)
    body = models.TextField()
    pic = models.ImageField(null=True, blank=True)
    date = models.DateTimeField(blank=True, null=True)
    