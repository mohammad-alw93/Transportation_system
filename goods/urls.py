from django.urls import path
from . import views


urlpatterns = [
    
    path('index',views.index,name='index'),
    path('index2',views.index2,name='index-2'),
    path('index3',views.Dealing_with_genetic_algorithm,name='index-3'),
 
]