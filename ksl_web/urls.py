from django.urls import path

from views import index, predict


urlpatterns = [
    path("", index, name="index"),
    path("api/predict/", predict, name="predict"),
]
