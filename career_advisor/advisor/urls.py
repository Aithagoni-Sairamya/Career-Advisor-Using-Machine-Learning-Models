from django.conf import settings
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('assessment/', views.skills_assessment, name='assessment'),
    path('job/<int:job_id>/', views.job_details, name='job_details'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),  # Redirect after logout
    path('signup/', views.signup, name='signup'),
    path('knowledge/', views.knowledge_page, name='knowledge_page'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('test/',views.mock_test, name='test'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
