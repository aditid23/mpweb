from django.contrib import admin
from django.urls import path
from . import views  

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'), 
    path('mpweb/', views.mpweb, name='mpweb'),
    path('solve-lp/', views.solve_lp, name='graphical_lp.html'),
    path('lpsimplex/', views.lpsimplex, name='lpsimplex'),
    path('solve_simplex/', views.solve_simplex, name='solve_simplex'),
    path('transportation/', views.transport, name='transportation'),  
    path('transport/', views.transportation_view, name='transport')
]