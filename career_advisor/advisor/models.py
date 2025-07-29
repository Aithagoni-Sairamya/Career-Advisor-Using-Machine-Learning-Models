from django.db import models
from django.contrib.auth.models import User

# Define choices for skill levels
SKILL_LEVEL_CHOICES = [
    (0, 'Not Interested'),
    (1, 'Poor'),
    (2, 'Beginner'),
    (3, 'Average'),
    (4, 'Intermediate'),
    (5, 'Professional'),
    (6, 'Excellent'),
]

class UserSkills(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    database_fundamentals = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    computer_architecture = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    distributed_computing_systems = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    cyber_security = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    networking = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    software_development = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    programming_skills = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    project_management = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    computer_forensics_fundamentals = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    technical_communication = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    ai_ml = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    software_engineering = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    business_analysis = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    communication_skills = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    data_science = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    troubleshooting_skills = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)
    graphics_designing = models.IntegerField(choices=SKILL_LEVEL_CHOICES, default=0)

    def get_feature_values(self):
        return [
            self.database_fundamentals, self.computer_architecture, self.distributed_computing_systems,
            self.cyber_security, self.networking, self.software_development, self.programming_skills,
            self.project_management, self.computer_forensics_fundamentals, self.technical_communication,
            self.ai_ml, self.software_engineering, self.business_analysis, self.communication_skills,
            self.data_science, self.troubleshooting_skills, self.graphics_designing
        ]
        
    def __str__(self):
        return f"{self.user.username}'s Skills"

class JobRole(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()
    skills_required = models.TextField()
    education_requirements = models.TextField()
    duties = models.TextField()
    min_salary = models.PositiveIntegerField()
    max_salary = models.PositiveIntegerField()
    companies = models.TextField()
    interview_qna = models.JSONField(default=list)  # Store questions and answers as JSON
    image = models.ImageField(upload_to='job_images/', null=True, blank=True)

    def __str__(self):
        return self.title

class Skill(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class UserJobRole(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    job_role = models.ForeignKey(JobRole, on_delete=models.CASCADE)

class Career(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.title
