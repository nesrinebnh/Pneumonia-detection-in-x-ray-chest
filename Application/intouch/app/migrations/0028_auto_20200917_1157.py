# Generated by Django 3.0.8 on 2020-09-17 10:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0027_auto_20200917_1142'),
    ]

    operations = [
        migrations.AlterField(
            model_name='test',
            name='date',
            field=models.DateField(auto_now_add=True),
        ),
    ]
