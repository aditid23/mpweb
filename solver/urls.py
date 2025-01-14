from django.contrib import admin
from django.urls import path
from . import views  # Import your views module

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),  # Route for the index page
    path('solve-lp/', views.solve_lp, name='solve_lp'),  # Route for the LP solver
]
