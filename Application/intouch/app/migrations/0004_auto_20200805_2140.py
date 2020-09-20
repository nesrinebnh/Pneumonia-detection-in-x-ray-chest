# Generated by Django 3.0.8 on 2020-08-05 20:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_auto_20200805_2021'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='patient',
            name='maladie_name',
        ),
        migrations.AlterField(
            model_name='maladie',
            name='maladie_name',
            field=models.CharField(blank=True, choices=[(None, 'Choose your gender'), ('male', 'male'), ('female', 'female'), ('custom', 'custom'), ('Prefer Not To Say', 'Prefer Not To Say')], max_length=200, verbose_name='gender'),
        ),
    ]
