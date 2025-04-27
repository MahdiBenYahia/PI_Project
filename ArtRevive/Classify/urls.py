from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('classify/', views.classify_view, name='classify'),  # URL for your model page
    path('predict/', views.predict, name='predict'),
]
