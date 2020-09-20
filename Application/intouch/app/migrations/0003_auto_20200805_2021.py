# Generated by Django 3.0.8 on 2020-08-05 19:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_auto_20200805_2004'),
    ]

    operations = [
        migrations.AddField(
            model_name='patient',
            name='maladie_name',
            field=models.CharField(blank=True, choices=[(None, 'Choose your gender'), ('male', 'male'), ('female', 'female'), ('custom', 'custom'), ('Prefer Not To Say', 'Prefer Not To Say')], max_length=200, verbose_name='gender'),
        ),
        migrations.AlterField(
            model_name='maladie',
            name='maladie_name',
            field=models.CharField(max_length=200, unique=True),
        ),
    ]
