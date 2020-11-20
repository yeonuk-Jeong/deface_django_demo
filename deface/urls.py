from django.urls import path
from django.views.generic import TemplateView

from . import views

urlpatterns = [
    path('', TemplateView.as_view(template_name='upload.html')),
    path('deface_nifti', views.deface, name='deface'),
]
