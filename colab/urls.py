from django.urls import path
from .views import *

app_name = 'colab'
urlpatterns = [
    path('grafico_1', grafico_1, name="grafico_1"),
    path('grafico_2', grafico_2, name="grafico_2"),
    path('grafico_3', grafico_3, name="grafico_3"),
    path('powerbi', powerbi, name='powerbi'),

]
