from .models import PatientProfile, Appointment, Doctor
from django.db.models import Count


def verification_status(request):
    """
    Adds to every template context:
      - user_id, name, role, email
      - patient_is_verified, verification_requested
      - pending_verify_count (for doctor badge)
      - pending_appt_count   (for doctor badge)
    """
    user_id = request.session.get('user_id')
    name = request.session.get('name', '')
    role = request.session.get('role', '')
    email = request.session.get('email', '')

    context = {
        'user_id': user_id,
        'name': name,
        'role': role,
        'email': email,
        'patient_is_verified': None,
        'verification_requested': False,
        'pending_verify_count': 0,
        'pending_appt_count': 0,
    }

    if not user_id:
        return context

    # Patient-specific
    if role == 'patient':
        try:
            profile = PatientProfile.objects.get(user_id=user_id)
            context['patient_is_verified'] = profile.is_verified
            context['verification_requested'] = profile.verification_requested
        except PatientProfile.DoesNotExist:
            context['patient_is_verified'] = False

    # Doctor-specific: badge counts
    if role == 'doctor':
        # Pending verifications (only real patients, only with scans)
        context['pending_verify_count'] = PatientProfile.objects.filter(
            is_verified=False,
            verification_requested=True,
            user__role='patient',
        ).annotate(scans_count=Count('user__detections')).filter(scans_count__gt=0).count()

        # Pending appointments for this doctor
        try:
            doctor = Doctor.objects.get(email=email)
            context['pending_appt_count'] = Appointment.objects.filter(
                doctor=doctor, status='pending'
            ).count()
        except Doctor.DoesNotExist:
            pass

    return context