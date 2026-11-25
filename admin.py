from django.contrib import admin
from .models import (
    User, Doctor, PatientProfile, Visit, Prescription,
    PatientDocument, MedicalCertificate, Appointment,
    DetectionHistory, PasswordResetRequest, Review,
)


# ============================================================
# CUSTOM BULK ACTIONS
# ============================================================

@admin.action(description='✅ Verify selected patients (mark as verified)')
def verify_patients_action(modeladmin, request, queryset):
    """Bulk-verify multiple patient profiles at once."""
    from django.utils import timezone
    count = queryset.filter(is_verified=False).update(
        is_verified=True,
        verified_at=timezone.now(),
        verified_by='admin-bulk-action',
    )
    modeladmin.message_user(request, f"{count} patient(s) verified.")


@admin.action(description='❌ Unverify selected patients')
def unverify_patients_action(modeladmin, request, queryset):
    count = queryset.filter(is_verified=True).update(
        is_verified=False,
        verified_at=None,
        verified_by='',
    )
    modeladmin.message_user(request, f"{count} patient(s) unverified.")


@admin.action(description='🔓 Reactivate selected users')
def activate_users_action(modeladmin, request, queryset):
    count = queryset.filter(is_active=False).update(is_active=True)
    modeladmin.message_user(request, f"{count} user(s) reactivated.")


@admin.action(description='🔒 Deactivate selected users')
def deactivate_users_action(modeladmin, request, queryset):
    # Don't let the current superuser deactivate themselves
    count = queryset.exclude(id=request.user.id).filter(is_active=True).update(is_active=False)
    modeladmin.message_user(request, f"{count} user(s) deactivated.")


@admin.action(description='🩺 Activate selected doctors')
def activate_doctors_action(modeladmin, request, queryset):
    count = queryset.filter(is_active=False).update(is_active=True)
    # Sync User login rows
    emails = list(queryset.values_list('email', flat=True))
    User.objects.filter(email__in=emails, role='doctor').update(is_active=True)
    modeladmin.message_user(request, f"{count} doctor(s) reactivated.")


@admin.action(description='🚫 Deactivate selected doctors')
def deactivate_doctors_action(modeladmin, request, queryset):
    count = queryset.filter(is_active=True).update(is_active=False)
    emails = list(queryset.values_list('email', flat=True))
    User.objects.filter(email__in=emails, role='doctor').update(is_active=False)
    modeladmin.message_user(request, f"{count} doctor(s) deactivated.")


@admin.action(description='✔ Mark selected appointments as Completed')
def complete_appointments_action(modeladmin, request, queryset):
    count = queryset.update(status='completed')
    modeladmin.message_user(request, f"{count} appointment(s) marked completed.")


@admin.action(description='✔ Mark selected appointments as Confirmed')
def confirm_appointments_action(modeladmin, request, queryset):
    count = queryset.update(status='confirmed')
    modeladmin.message_user(request, f"{count} appointment(s) confirmed.")


# ============================================================
# INLINE FORMS
# ============================================================

class PrescriptionInline(admin.TabularInline):
    """Show prescriptions inline when editing a Visit."""
    model = Prescription
    extra = 1
    fields = ('medication', 'dose', 'frequency', 'duration', 'instructions')


class VisitInline(admin.StackedInline):
    """Show visits inline when editing a PatientProfile."""
    model = Visit
    extra = 0
    fields = ('visit_date', 'doctor', 'reason', 'diagnosis', 'follow_up_date')
    show_change_link = True


class PatientDocumentInline(admin.TabularInline):
    """Show documents inline when editing a PatientProfile."""
    model = PatientDocument
    extra = 0
    fields = ('doc_type', 'title', 'file', 'uploaded_by')
    show_change_link = True


class MedicalCertificateInline(admin.TabularInline):
    """Show certificates inline when editing a PatientProfile."""
    model = MedicalCertificate
    extra = 0
    fields = ('doctor', 'issued_date', 'diagnosis', 'rest_days', 'valid_until')
    show_change_link = True


# ============================================================
# MODEL ADMINS
# ============================================================

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'role', 'is_active', 'created_at')
    list_filter = ('role', 'is_active')
    search_fields = ('name', 'email')
    ordering = ('-id',)
    actions = [activate_users_action, deactivate_users_action]


@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'specialization', 'is_active')
    list_filter = ('is_active', 'specialization')
    search_fields = ('name', 'email', 'specialization')
    ordering = ('-id',)
    actions = [activate_doctors_action, deactivate_doctors_action]


@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = (
        'patient_id', 'user', 'is_verified', 'verification_requested',
        'phone', 'city', 'registration_date',
    )
    list_filter = ('is_verified', 'verification_requested', 'gender', 'smoker')
    search_fields = ('patient_id', 'user__name', 'user__email', 'phone', 'cnic')
    readonly_fields = ('registration_date', 'updated_at')
    ordering = ('-id',)
    actions = [verify_patients_action, unverify_patients_action]

    # INLINES — edit a patient and see their visits/documents/certificates inline
    inlines = [VisitInline, PatientDocumentInline, MedicalCertificateInline]

    fieldsets = (
        ('Identity', {
            'fields': ('user', 'patient_id', 'phone', 'date_of_birth', 'gender', 'cnic')
        }),
        ('Address', {
            'fields': ('address', 'city')
        }),
        ('Verification', {
            'fields': ('is_verified', 'verified_at', 'verified_by',
                       'verification_requested', 'verification_requested_at'),
        }),
        ('Medical History', {
            'fields': ('chronic_conditions', 'allergies', 'current_medications',
                       'past_surgeries', 'smoker', 'pregnant'),
        }),
        ('Insurance & Emergency', {
            'fields': ('insurance_provider', 'insurance_number',
                       'emergency_contact_name', 'emergency_contact_phone'),
        }),
        ('Timestamps', {
            'fields': ('registration_date', 'updated_at'),
        }),
    )


@admin.register(Visit)
class VisitAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'doctor', 'visit_date', 'reason', 'follow_up_date')
    list_filter = ('visit_date', 'doctor')
    search_fields = ('patient__patient_id', 'patient__user__name', 'reason', 'diagnosis')
    ordering = ('-visit_date',)

    # INLINE — edit a visit and its prescriptions on the same page
    inlines = [PrescriptionInline]

    fieldsets = (
        ('Context', {
            'fields': ('patient', 'doctor', 'visit_date', 'follow_up_date')
        }),
        ('Clinical Notes', {
            'fields': ('reason', 'chief_complaint', 'examination_findings',
                       'diagnosis', 'treatment_plan', 'notes'),
        }),
    )


@admin.register(Prescription)
class PrescriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'visit', 'medication', 'dose', 'frequency', 'duration')
    search_fields = ('medication', 'visit__patient__patient_id')
    ordering = ('-id',)


@admin.register(PatientDocument)
class PatientDocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'doc_type', 'title', 'uploaded_at', 'uploaded_by')
    list_filter = ('doc_type', 'uploaded_at')
    search_fields = ('title', 'patient__patient_id', 'patient__user__name')
    ordering = ('-uploaded_at',)


@admin.register(MedicalCertificate)
class MedicalCertificateAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'doctor', 'issued_date', 'rest_days', 'valid_until')
    list_filter = ('issued_date', 'doctor')
    search_fields = ('patient__patient_id', 'patient__user__name', 'diagnosis')
    ordering = ('-issued_date',)


@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'patient', 'doctor', 'appointment_date',
        'appointment_time', 'status', 'created_at',
    )
    list_filter = ('status', 'appointment_date', 'doctor')
    search_fields = ('patient__name', 'patient__email', 'doctor__name', 'reason')
    ordering = ('-appointment_date', '-appointment_time')
    actions = [confirm_appointments_action, complete_appointments_action]


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'patient', 'detected_diseases', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('user__name', 'user__email', 'detected_diseases')
    ordering = ('-timestamp',)


@admin.register(PasswordResetRequest)
class PasswordResetRequestAdmin(admin.ModelAdmin):
    list_display = ('id', 'email', 'used', 'created_at')
    list_filter = ('used', 'created_at')
    search_fields = ('email',)
    ordering = ('-created_at',)


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'doctor', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('user__name', 'user__email', 'comment')
    ordering = ('-created_at',)