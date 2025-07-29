from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import UserSkills, JobRole, Career, SKILL_LEVEL_CHOICES
from .ml_model import predict_careers
from django.http import Http404
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

def home(request):
    return render(request, 'advisor/home.html')

def about(request):
    return render(request, 'advisor/about.html')

def contact(request):
    return render(request, 'advisor/contact.html')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to login page after successful signup
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "Logged in successfully!")
            return redirect('home')  # Redirect to the home page or dashboard
        else:
            messages.error(request, "Invalid username or password. Please try again.")
    else:
        form = AuthenticationForm()

    return render(request, 'registration/login.html', {'form': form})

@login_required
def skills_assessment(request):
    if request.method == 'POST':
        # Create UserSkills object from form data
        skills = UserSkills(
            user=request.user,
            database_fundamentals=int(request.POST.get('database_fundamentals', 0)),
            computer_architecture=int(request.POST.get('computer_architecture', 0)),
            distributed_computing_systems=int(request.POST.get('distributed_computing_systems', 0)),
            cyber_security=int(request.POST.get('cyber_security', 0)),
            networking=int(request.POST.get('networking', 0)),
            software_development=int(request.POST.get('software_development', 0)),
            programming_skills=int(request.POST.get('programming_skills', 0)),
            project_management=int(request.POST.get('project_management', 0)),
            computer_forensics_fundamentals=int(request.POST.get('computer_forensics_fundamentals', 0)),
            technical_communication=int(request.POST.get('technical_communication', 0)),
            ai_ml=int(request.POST.get('ai_ml', 0)),
            software_engineering=int(request.POST.get('software_engineering', 0)),
            business_analysis=int(request.POST.get('business_analysis', 0)),
            communication_skills=int(request.POST.get('communication_skills', 0)),
            data_science=int(request.POST.get('data_science', 0)),
            troubleshooting_skills=int(request.POST.get('troubleshooting_skills', 0)),
            graphics_designing=int(request.POST.get('graphics_designing', 0))
        )
        skills.save()

        # Get career predictions
        predictions = predict_careers(skills)

        # Get job roles in the order of predictions
        jobs = [JobRole.objects.get(title=role) for role in predictions]

        context = {
            'predictions': predictions,
            'jobs': jobs
        }

        return render(request, 'advisor/results.html', context)

    return render(request, 'advisor/assessment.html', {'SKILL_LEVEL_CHOICES': SKILL_LEVEL_CHOICES})

def knowledge_page(request):
    """
    View to display the knowledge page with career options, job roles, and relevant information.
    """
    try:
        careers = Career.objects.all()
        job_roles = JobRole.objects.all()
    except Career.DoesNotExist or JobRole.DoesNotExist:
        raise Http404("Knowledge content not found")

    context = {
        'careers': careers,
        'job_roles': job_roles,
        'skill_levels': SKILL_LEVEL_CHOICES,  
    }

    return render(request, 'advisor/knowledge_page.html', context)

def job_details(request, job_id):
    try:
        job = JobRole.objects.get(id=job_id)
        return render(request, 'advisor/job_details.html', {'job': job})
    except JobRole.DoesNotExist:
        return redirect('home')
def feedback_view(request):
    return render(request, 'advisor/feedback.html')

def mock_test(request):
    return render(request, 'advisor/mock_test.html')
