# Generated by Django 3.0.8 on 2020-08-07 00:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0023_message_title'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='date',
            field=models.DateField(blank=True, null=True),
        ),
    ]