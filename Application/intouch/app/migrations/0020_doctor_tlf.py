# Generated by Django 3.0.8 on 2020-08-06 18:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0019_auto_20200806_1351'),
    ]

    operations = [
        migrations.AddField(
            model_name='doctor',
            name='tlf',
            field=models.CharField(max_length=200, null=True),
        ),
    ]