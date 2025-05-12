from django.urls import path
from .views import ArtQueryView, CheckInitView

urlpatterns = [
    path('api/query/', ArtQueryView.as_view(), name='art-query'),
    path('api/check_init/', CheckInitView.as_view()),
]