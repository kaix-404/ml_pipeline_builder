from django.urls import path
from django.shortcuts import render
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_page, name='upload'),
    path('preprocess/', views.preprocess_page, name='preprocess'),
    path("preprocess/extract-text/", views.extract_text, name="extract_text"),
    path("preprocess/standardize/", views.standardize, name="standardize"),
    path("preprocess/normalize/", views.normalize, name="normalize"),
    path('split/', views.split_page, name='split'),
    path('apply-split/', views.train_test_split_view, name='apply_split'),
    path('model/', lambda request: render(request, 'model.html'), name='model_page'),
    path('train_model/', views.train_model, name='train_model'),
    path('results/', views.results, name='results'),
]

