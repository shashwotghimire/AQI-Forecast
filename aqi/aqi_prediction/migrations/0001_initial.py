# Generated by Django 5.0.7 on 2024-11-05 06:18

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AQIData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('datetime', models.DateTimeField()),
                ('pm25', models.FloatField()),
                ('o3', models.FloatField()),
            ],
            options={
                'ordering': ['datetime'],
            },
        ),
    ]
