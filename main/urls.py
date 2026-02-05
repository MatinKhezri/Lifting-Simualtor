# main/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("logout/", views.user_logout, name="logout"),  # ← اصلاح شد
    path("accounts/login/", views.user_login, name="login"),
    path("api/predict_all/", views.api_predict_all, name="api_predict_all"),
    path("api/predict_markers/", views.api_predict_markers, name="api_predict_markers"),
    path("api/predict_dynamic/", views.api_predict_dynamic, name="api_predict_dynamic"),

]
