from django.db import models
from django.utils import timezone
from datetime import timedelta
import random
import string


class User(models.Model):
    ROLE_CHOICES = [
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
        ('admin', 'Admin'),
    ]
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='patient')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} ({self.email}) — {self.role}"


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    specialization = models.CharField(max_length=100, blank=True)
    available_days = models.CharField(max_length=200, blank=True)
    available_times = models.CharField(max_length=200, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} — {self.specialization}"


class PatientProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='patient_profile'
    )
    patient_id = models.CharField(max_length=20, unique=True, blank=True)
    is_verified = models.BooleanField(default=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    verified_by = models.CharField(max_length=100, blank=True)
    verification_requested = models.BooleanField(default=False)
    verification_requested_at = models.DateTimeField(null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True)
    cnic = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    city = models.CharField(max_length=100, blank=True)
    chronic_conditions = models.TextField(blank=True)
    allergies = models.TextField(blank=True)
    current_medications = models.TextField(blank=True)
    past_surgeries = models.TextField(blank=True)
    smoker = models.BooleanField(default=False)
    pregnant = models.BooleanField(default=False)
    insurance_provider = models.CharField(max_length=100, blank=True)
    insurance_number = models.CharField(max_length=50, blank=True)
    emergency_contact_name = models.CharField(max_length=100, blank=True)
    emergency_contact_phone = models.CharField(max_length=20, blank=True)
    registration_date = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.patient_id:
            self.patient_id = self._generate_patient_id()
        super().save(*args, **kwargs)

    @staticmethod
    def _generate_patient_id():
        year = timezone.now().year
        last = PatientProfile.objects.filter(
            patient_id__startswith=f"PT-{year}-"
        ).order_by('-id').first()
        if last and last.patient_id:
            try:
                last_num = int(last.patient_id.split('-')[-1])
            except (ValueError, IndexError):
                last_num = 0
        else:
            last_num = 0
        return f"PT-{year}-{last_num + 1:04d}"

    @property
    def scan_count(self):
        return self.user.detections.count() if self.user_id else 0

    def __str__(self):
        return f"{self.patient_id} — {self.user.name}"


class Visit(models.Model):
    patient = models.ForeignKey(
        PatientProfile, on_delete=models.CASCADE, related_name='visits'
    )
    doctor = models.ForeignKey(
        Doctor, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='visits'
    )
    visit_date = models.DateTimeField(default=timezone.now)
    reason = models.CharField(max_length=200, blank=True)
    chief_complaint = models.TextField(blank=True)
    examination_findings = models.TextField(blank=True)
    diagnosis = models.TextField(blank=True)
    treatment_plan = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    follow_up_date = models.DateField(null=True, blank=True)

    class Meta:
        ordering = ['-visit_date']

    def __str__(self):
        return f"Visit {self.id} — {self.patient.patient_id} on {self.visit_date:%Y-%m-%d}"


class Prescription(models.Model):
    visit = models.ForeignKey(
        Visit, on_delete=models.CASCADE, related_name='prescriptions'
    )
    medication = models.CharField(max_length=100)
    dose = models.CharField(max_length=50, blank=True)
    frequency = models.CharField(max_length=50, blank=True)
    duration = models.CharField(max_length=50, blank=True)
    instructions = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.medication} — {self.dose}"


class PatientDocument(models.Model):
    DOC_TYPES = [
        ('xray', 'X-Ray'),
        ('lab', 'Lab Report'),
        ('consent', 'Consent Form'),
        ('photo', 'Clinical Photo'),
        ('report', 'Report'),
        ('other', 'Other'),
    ]
    patient = models.ForeignKey(
        PatientProfile, on_delete=models.CASCADE, related_name='documents'
    )
    doc_type = models.CharField(max_length=20, choices=DOC_TYPES, default='other')
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='patient_docs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.CharField(max_length=100, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.title} ({self.get_doc_type_display()})"


class MedicalCertificate(models.Model):
    patient = models.ForeignKey(
        PatientProfile, on_delete=models.CASCADE, related_name='certificates'
    )
    doctor = models.ForeignKey(
        Doctor, on_delete=models.SET_NULL, null=True, blank=True
    )
    issued_date = models.DateTimeField(default=timezone.now)
    diagnosis = models.TextField()
    recommendation = models.TextField(blank=True)
    rest_days = models.PositiveIntegerField(default=0)
    valid_until = models.DateField(null=True, blank=True)

    class Meta:
        ordering = ['-issued_date']

    def __str__(self):
        return f"Certificate #{self.id} — {self.patient.patient_id}"


class Appointment(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    patient = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='appointments'
    )
    doctor = models.ForeignKey(
        Doctor, on_delete=models.CASCADE, related_name='appointments'
    )
    appointment_date = models.DateField()
    appointment_time = models.TimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    reason = models.CharField(max_length=200, blank=True)
    doctor_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-appointment_date']

    def __str__(self):
        return f"{self.patient.name} → {self.doctor.name} on {self.appointment_date}"


class DetectionHistory(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='detections',
        null=True, blank=True
    )
    patient = models.ForeignKey(
        PatientProfile, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='scans'
    )
    image_path = models.CharField(max_length=500)
    detected_diseases = models.CharField(max_length=500, blank=True)
    confidence_scores = models.CharField(max_length=500, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        uname = self.user.name if self.user else "Unknown"
        return f"Scan #{self.id} — {uname} — {self.detected_diseases}"


class PasswordResetRequest(models.Model):
    email = models.EmailField()
    code = models.CharField(max_length=10, blank=True, default='')
    used = models.BooleanField(default=False)
    used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        state = "used" if self.used else "pending"
        return f"Reset for {self.email} — {state}"

    @staticmethod
    def generate_code():
        return ''.join(random.choices(string.digits, k=6))

    def is_expired(self):
        if not self.created_at:
            return False
        return (timezone.now() - self.created_at) > timedelta(minutes=15)


class Review(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='reviews'
    )
    doctor = models.ForeignKey(
        Doctor, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='reviews'
    )
    rating = models.PositiveSmallIntegerField(default=5)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Review by {self.user.name} — {self.rating}/5"


# ============== AUDIT LOG (Task I) ==============

class AuditLog(models.Model):
    """
    Immutable record of security-relevant and clinical-relevant actions.
    Used for compliance, debugging, and clinical accountability.
    """
    ACTION_CHOICES = [
        ('login', 'User Login'),
        ('login_failed', 'Failed Login Attempt'),
        ('logout', 'User Logout'),
        ('register', 'New User Registration'),
        ('password_reset', 'Password Reset Completed'),
        ('scan_created', 'AI Scan Performed'),
        ('report_downloaded', 'Report PDF Downloaded'),
        ('certificate_issued', 'Medical Certificate Issued'),
        ('certificate_downloaded', 'Certificate Downloaded'),
        ('visit_added', 'Clinical Visit Added'),
        ('prescription_added', 'Prescription Added'),
        ('document_uploaded', 'Patient Document Uploaded'),
        ('patient_verified', 'Patient Verified'),
        ('patient_unverified', 'Patient Verification Revoked'),
        ('appointment_booked', 'Appointment Booked'),
        ('appointment_status_changed', 'Appointment Status Changed'),
        ('appointment_cancelled', 'Appointment Cancelled'),
        ('review_submitted', 'Review Submitted'),
        ('review_deleted', 'Review Deleted'),
        ('user_role_changed', 'User Role Changed'),
        ('user_deactivated', 'User Deactivated'),
        ('user_reactivated', 'User Reactivated'),
        ('doctor_added', 'Doctor Added'),
        ('doctor_edited', 'Doctor Edited'),
        ('doctor_deactivated', 'Doctor Deactivated'),
        ('doctor_reactivated', 'Doctor Reactivated'),
        ('other', 'Other'),
    ]

    # Who
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='audit_logs'
    )
    user_email = models.CharField(max_length=200, blank=True)
    user_role = models.CharField(max_length=20, blank=True)

    # What
    action = models.CharField(max_length=50, choices=ACTION_CHOICES, default='other')
    description = models.CharField(max_length=500, blank=True)

    # On what
    target_type = models.CharField(max_length=50, blank=True)
    target_id = models.CharField(max_length=50, blank=True)

    # Context
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=300, blank=True)

    # When
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Audit Log Entry'
        verbose_name_plural = 'Audit Logs'

    def __str__(self):
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.user_email} — {self.get_action_display()}"

    @property
    def action_color(self):
        """Bootstrap color code for the badge."""
        if self.action in ('login', 'patient_verified', 'user_reactivated',
                           'doctor_reactivated', 'review_submitted',
                           'certificate_issued', 'scan_created', 'visit_added',
                           'prescription_added', 'document_uploaded', 'register'):
            return 'success'
        if self.action in ('login_failed', 'user_deactivated', 'patient_unverified',
                           'doctor_deactivated', 'review_deleted',
                           'appointment_cancelled'):
            return 'danger'
        if self.action in ('logout', 'appointment_status_changed',
                           'user_role_changed', 'appointment_booked'):
            return 'warning'
        return 'info'