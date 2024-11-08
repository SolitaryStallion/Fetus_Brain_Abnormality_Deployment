from django.urls import path
from . import views  # import your views

urlpatterns = [
    path('upload/', views.upload_image , name='upload'),  # URL for the file upload page
    path('meow/', views.meow)
]