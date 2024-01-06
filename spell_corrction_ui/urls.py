from django.urls import path
from . import views

urlpatterns = [
    path('secondpage/', views.secondpage)
]
