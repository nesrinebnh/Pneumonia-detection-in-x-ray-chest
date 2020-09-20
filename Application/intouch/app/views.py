from django.shortcuts import render
from django.utils import timezone
from django.shortcuts import render, get_object_or_404
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import Group
from django.db.models import Q

import pickle

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras import backend as K

import tensorflow as tf
import json
from tensorflow import Graph




from io import BytesIO, StringIO
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.template.loader import render_to_string
from django.template import RequestContext

import os

from .forms import *
from .models import *

from .decorators import *

from django.core.files.storage import FileSystemStorage
# Create your views here.

@unauthenticated_user
def homePage(request):
    if request.method == 'POST':
        return redirect('choice')
    return render(request, 'app/home.html', {})

@unauthenticated_user
def choice(request):
    if request.method == 'POST':
        global choice
        if 'medecin' in request.POST:
            choice = 'doctor'
        elif 'malade' in request.POST:  
            choice = 'patient'
        else: 
            if isinstance(choice, str):
                form = createUserForm()
                context = {'choice': choice, 'form':form}
                return render(request, 'registration/signup.html', context)  
            else:
                messages.success(request, 'Vous devez choisir un des options pour continuer ')
                return redirect('choice')
    
    return render(request, 'app/patient_or_doctor.html', {})


@unauthenticated_user
def registerPatient(request):
    print(request.user)
    form_patient = PatientForm()
    form_malade = maladieForm(prefix="maladie")
    if request.method == 'POST':
        form_patient = PatientForm(request.POST)
        if form_patient.is_valid():

            print(form_patient.cleaned_data.get('name'))
            print(form_patient.cleaned_data.get('prenom'))
            print(form_patient.cleaned_data.get('age'))
            patient = form_patient.save()
            return render(request, 'app/patient_maladie.html', {'pk':patient.pk,'maladie':form_malade})
    return render(request, 'app/patient_register.html',{'patient':form_patient})


@unauthenticated_user
def register_maladie(request, pk):
    form_malade = maladieForm()
    return render(request, 'app/patient_maladie.html', {'pk':pk,'maladie':form_malade})


@unauthenticated_user
def registerPage(request, pk_test):
    
    form = createUserForm()
    if request.method == 'POST':
        form = createUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            group = Group.objects.get(name=pk_test)
            user.groups.add(group)
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(request=request, username= username, password=password)
            if user is not None:
                login(request, user)
                return redirect('register_patient')
    return render(request, 'registration/signup.html',{'form':form})



@unauthenticated_user
def login_page(request):
   
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request=request, username= username, password=password)

        if user is not None:
            if user.groups.filter(name='patient').count():
                print("ok")
                login(request, user)
                return redirect('patient_home')
        else:
            messages.info(request, 'Username Or password is incorrect')
    return render(request, 'registration/login.html', {})

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def patient_home(request):
    patient = Patient.objects.get(user=request.user.pk)
    maladie = Maladie.objects.all().filter(patient=patient)
    return render(request, 'app/patient.html',{'patient':patient, 'maladie':maladie})

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def mesmedecin(request, pk_mesmedecin):
    patient = Patient.objects.get(user=request.user)
    doctors = Doctor.objects.all().filter(patient=patient)
    return render(request, 'app/mesmedecins.html',{'doctor': doctors})

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def contact(request, pk_patient, pk_medecin):
    form = MessageForm()
    doctor = Doctor.objects.get(pk=pk_medecin)
    if request.method == 'POST':
        if 'Envoyermessage' in request.POST:
            form = MessageForm(request.POST, request.FILES)
            if form.is_valid():
                msg = form.save(commit=False)
                msg.sender = request.user.username
                msg.receiver = doctor.user.username
                msg.date = timezone.now()
                msg.save()
                patient = Patient.objects.get(user=request.user)
                doctors = Doctor.objects.all().filter(patient=patient)
                return render(request, 'app/mesmedecins.html',{'doctor': doctors})
    
    return render(request, 'app/contacter_medecin.html',{'form': form, 'doctor': doctor})

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def mesMessages(request, pk_patient,medecin):
    print(request.user.pk)
    doctor = Doctor.objects.get(pk=medecin)
    msgsend = Message.objects.filter(Q(date__lte=timezone.now()) & (Q(sender__exact=request.user.username) & Q(receiver__exact=doctor.user.username)) | (Q(receiver__exact=request.user.username)& Q(sender__exact=doctor.user.username))).order_by('-date')
    return render(request, 'app/mes_messages.html',{'msgsend':msgsend, 'medecin':doctor})

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def messagedetail(request, pk_patient,medecin, pk_message):
    message = Message.objects.get(pk=pk_message)
    return render(request, 'app/message_detail.html',{'msg': message})


def lougout_user(request):
    logout(request)
    return redirect('login')


def doctorPage(request, pk_doctor):
	context = {}
	return render(request, 'app/doctor.html', context)

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def patientPage(request, pk_patient):
    patient = Patient.objects.get(user=request.user.pk)
    maladie = Maladie.objects.all().filter(patient=patient)
    context = {'patient':patient,'maladie':maladie}
    return render(request, 'app/patient.html', context)

@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def Radio(request, pk_patient):
    patient = Patient.objects.get(user=request.user.pk)
    test = Test.objects.all().filter(patient=patient)
    return render(request, 'app/mes_radio.html',{'test': test})


@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def AddRadio(request, pk_patient):
    form = TestForm()
    if request.method == 'POST':
        if 'Diagnostic' in request.POST:
            form = TestForm(request.POST, request.FILES)
            if form.is_valid():
                patient = Patient.objects.get(user=request.user.pk)
                saved = form.save(commit=False)
                saved.patient = patient
                saved.save() 
                
                print(saved.test_image.path)
                img_height, img_width=150,150
                with open('./model/pneumonia_model.json','r') as f:
                    labelInfo=f.read()

                labelInfo=json.loads(labelInfo)


                model_graph = Graph()
                with model_graph.as_default():
                    model=load_model('./model/pneumonia_model.h5')

                IMG = image.load_img(saved.test_image.path).convert('L')
                IMG_ = IMG.resize((150, 150))
                IMG_ = np.asarray(IMG_)
                IMG_ = np.true_divide(IMG_, 255)
                IMG_ = IMG_.reshape(1, 150, 150, 1)

                with model_graph.as_default():
                    
                    predi=model.predict(IMG_)
                    label = model.predict_classes(IMG_)
                classe = ['pneumonia', 'normal']
                print(classe[(label[0])[0]])
                probability = "Le résultat du diagnostic : "+classe[(label[0])[0]]
                saved.probability = probability
                saved.save()

                return render(request, 'app/add_radio.html',{'form':form, 'saved':saved})
    return render(request, 'app/add_radio.html',{'form':form})


def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html  = template.render(context_dict)
    response = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), response)
    if not pdf.err:
        return HttpResponse(response.getvalue(), content_type='application/pdf')
    else:
        return HttpResponse("Error Rendering PDF", status=400)


@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def ViewPDF(request,pk_patient, pk):
    saved = Test.objects.get(pk=pk)
    user = Patient.objects.get(user = request.user)
    
    pdf = render_to_pdf('app/pdf.html', {'saved': saved, 'user':user, 'time': timezone.now()})
    return HttpResponse(pdf, content_type='application/pdf')

#Automaticly downloads to PDF file
@login_required(login_url='login')
@allowed_users(allowed_roles=['patient'])
def DownloadPDF(request,pk_patient, pk):
    print(pk)
    saved = Test.objects.get(pk=pk)
    pdf = render_to_pdf('app/pdf.html', {'saved': saved})

    response = HttpResponse(pdf, content_type='application/pdf')
    filename = "résultat_du_billan.pdf"
    content = "attachment; filename='%s'" %(filename)
    response['Content-Disposition'] = content
    return response
