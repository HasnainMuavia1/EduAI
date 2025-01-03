"""
URL configuration for EduAI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path("",views.transcribe, name="Transcribe"),
    path("transcribe/", views.trans_results, name="trans_results"),
    path("chatWithBook/",views.chat_with_book,name="chat_with_book"),
    path('pdf/', views.chat, name='chat'),
    path('ocr/', views.ocr, name='ocr'),
]
