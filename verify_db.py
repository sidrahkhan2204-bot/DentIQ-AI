import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dental_project.settings')
django.setup()

from ai_app.models import (
    User, Doctor, Appointment, DetectionHistory,
    PasswordResetRequest, Review,
)

print("=" * 50)
print("DATABASE VERIFICATION")
print("=" * 50)
print("Users:            ", User.objects.count())
print("Doctors:          ", Doctor.objects.count())
print("Appointments:     ", Appointment.objects.count())
print("Scans:            ", DetectionHistory.objects.count())
print("Reset requests:   ", PasswordResetRequest.objects.count())
print("Reviews:          ", Review.objects.count())
print()
print("Sample user:      ", User.objects.first())
print("Sample doctor:    ", Doctor.objects.first())

d = Doctor.objects.first()
if d:
    print("Doctor city:      ", d.city)
    print("Avg rating:       ", d.average_rating())
    print("Review count:     ", d.review_count())