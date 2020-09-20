from django.contrib import admin
from app.models import Patient, Maladie, Doctor, Test, Message
# Register your models here.

admin.site.register(Patient)
admin.site.register(Maladie)
admin.site.register(Doctor)
admin.site.register(Test)
admin.site.register(Message)