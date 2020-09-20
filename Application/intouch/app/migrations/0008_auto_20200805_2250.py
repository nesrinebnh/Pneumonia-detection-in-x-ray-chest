# Generated by Django 3.0.8 on 2020-08-05 21:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0007_doctor'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='doctor',
            name='patient',
        ),
        migrations.AddField(
            model_name='doctor',
            name='patient',
            field=models.ManyToManyField(to='app.Patient'),
        ),
    ]