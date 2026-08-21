from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('start/', views.start_whiteboard),
]