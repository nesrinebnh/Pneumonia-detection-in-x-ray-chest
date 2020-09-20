# Generated by Django 3.0.8 on 2020-08-05 21:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0009_auto_20200805_2256'),
    ]

    operations = [
        migrations.AlterField(
            model_name='maladie',
            name='maladie_name',
            field=models.CharField(blank=True, choices=[(None, 'Liste de maladie chronique'), ('Diabete', 'Diabete'), ('Hypertension arterielle', 'Hypertension arterielle'), ('Maladies rhumatismale', 'Maladies rhumatismale'), ('Maladies cardiaque', 'Maladies cardiaque')], max_length=200, unique=True, verbose_name='gender'),
        ),
    ]
