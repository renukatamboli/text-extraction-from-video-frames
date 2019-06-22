

from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [

	path('',views.video,name='video'),
	path('UploadImage/',views.examples,name='UploadImage'),
	path('AboutUs/',views.AboutUs,name='aboutus')

]
