from django.urls import path
from . import views

urlpatterns = [
   
    path('', views.homePage, name='home'),
    path('register/', views.registerPage, name='register'),
    path('login/', views.login_page, name='login'),
    path('logout/', views.lougout_user, name='logout'),

    #register patient urls
    path('register/choice', views.choice, name='choice'),
    path('register/<str:pk_test>/', views.registerPage, name='doctor_or_patient'),
    path('register/info',views.registerPatient, name='register_patient'),
    path('register/info/<int:pk>', views.register_maladie, name='register_maladie'),

    #patient urls
    path('patient/', views.patient_home, name='patient_home'), 
    path('patient/<str:pk_patient>/', views.patientPage, name="patient"),
    path('patient/<str:pk_patient>/radio/', views.Radio, name="radio"),
    path('patient/<str:pk_patient>/ajouterradio/', views.AddRadio, name="add_radio"),
    path('patient/<str:pk_patient>/ajouterradio/download/<int:pk>/', views.ViewPDF, name="download"),
    path('patient/<str:pk_mesmedecin>/mesmedecins/', views.mesmedecin, name="mesMedecin"),
    path('patient/<str:pk_patient>/mesmedecins/contact/<int:pk_medecin>/newmessage/', views.contact, name="contact"),
    path('patient/<str:pk_patient>/mesmedecins/contact/<int:medecin>/', views.mesMessages, name="mesMessages"),
    path('patient/<str:pk_patient>/mesmedecins/contact/<int:medecin>/<int:pk_message>/', views.messagedetail, name="messagedetail"),

    #doctor urls
    path('doctor/<str:pk_doctor>/', views.doctorPage, name="doctor"),
]