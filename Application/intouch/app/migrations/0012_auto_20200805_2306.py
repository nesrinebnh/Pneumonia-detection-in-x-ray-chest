# Generated by Django 3.0.8 on 2020-08-05 22:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_doctor_user'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='doctor',
            name='patient',
        ),
        migrations.AddField(
            model_name='doctor',
            name='patient',
            field=models.ManyToManyField(null=True, to='app.Patient'),
        ),
    ]
