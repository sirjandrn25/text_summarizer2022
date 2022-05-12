from django.urls import path

from summarizer.views import HomeView

urlpatterns = [
    path("",HomeView.as_view(),name="index"),
]
