import os
import re
import cv2
import shutil
import io
import secrets
import string
import random
import numpy as np
import torch
import torch.nn.functional as F
import base64
from collections import Counter
from datetime import datetime, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse, FileResponse, Http404, HttpResponse
from django.db.models import Q, Count, Avg
from django.utils import timezone
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# NEW IMPORT FOR TEETH CHECK
from transformers import pipeline

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle,
)
from PIL import Image as PILImage

from .models import (
    User, Doctor, Appointment, DetectionHistory,
    PatientProfile, Visit, Prescription, PatientDocument, MedicalCertificate,
    PasswordResetRequest, Review, AuditLog,
)

BASE_DIR = settings.BASE_DIR

# ============== LOAD AI MODELS ==============
print("Loading AI models...")

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=os.path.join(BASE_DIR, 'models', 'detect_augmented_final.pt'),
    confidence_threshold=0.4,
    device='cpu'
)

segmentation_model = YOLO(os.path.join(BASE_DIR, 'models', 'seg_augmented_final.pt'))

try:
    print("Detection model class names:", detection_model.category_mapping)
except Exception:
    pass
try:
    print("Segmentation model class names:", segmentation_model.names)
except Exception:
    pass

print("All models loaded successfully! (confidence threshold = 0.4)")

# ============== CONSTANTS ==============
CONF_THRESHOLD = 0.40

IRRELEVANT_MSG = (
    "This image does not appear to be a valid dental close-up. "
    "Please upload a clear, close-up colour photo of your teeth."
)

XRAY_MSG = (
    "This app analyses colour dental photographs only. "
    "Please upload an intraoral colour photograph, not a radiograph or X-ray."
)

ALLOWED_CLASSES = {'calculus', 'caries', 'tooth_discoloration', 'hypodontia'}

OUT_OF_SCOPE_DISPLAY = {
    'gingivitis': 'gingivitis (gum inflammation)',
    'ulcer': 'oral ulcer',
}

SUGGESTION_MAP = {
    'Caries / Discoloration': {
        'high':   ("Schedule a dental appointment soon. Caries will not heal on its own and can progress if untreated.", 'High', '#DC2626'),
        'medium': ("Consult a dentist to confirm the finding. Early intervention prevents progression.", 'Medium', '#D97706'),
        'low':    ("Monitor the area and consult a dentist if symptoms develop.", 'Low', '#F59E0B'),
    },
    'Calculus': {
        'high':   ("Professional cleaning (scaling and polishing) is recommended to remove calculus build-up.", 'Medium', '#D97706'),
        'medium': ("Consider a professional dental cleaning at your next checkup.", 'Low', '#F59E0B'),
        'low':    ("Maintain regular brushing and flossing. Consult a dentist if plaque persists.", 'Low', '#F59E0B'),
    },
    'Tooth Discoloration': {
        'high':   ("Consider professional whitening or a cosmetic dentistry consultation.", 'Low', '#F59E0B'),
        'medium': ("Discuss whitening options with your dentist at your next visit.", 'Low', '#F59E0B'),
        'low':    ("Continue regular oral hygiene. Monitor for changes.", 'Routine', '#059669'),
    },
    'Hypodontia': {
        'high':   ("Consult an orthodontist. Missing teeth can affect alignment and chewing.", 'Medium', '#D97706'),
        'medium': ("Discuss treatment options with a dental specialist.", 'Medium', '#D97706'),
        'low':    ("Mention at your next dental visit for professional assessment.", 'Low', '#F59E0B'),
    },
}

# ============== AUDIT LOG HELPER ==============

def log_action(request, action, description='', target_type='', target_id=''):
    try:
        user_id = request.session.get('user_id') if hasattr(request, 'session') else None
        user_obj = None
        user_email = ''
        user_role = ''
        if user_id:
            user_obj = User.objects.filter(id=user_id).first()
            if user_obj:
                user_email = user_obj.email
                user_role = user_obj.role

        ip = None
        try:
            x_forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded:
                ip = x_forwarded.split(',')[0].strip()
            else:
                ip = request.META.get('REMOTE_ADDR')
        except Exception:
            pass

        ua = request.META.get('HTTP_USER_AGENT', '')[:300] if hasattr(request, 'META') else ''

        AuditLog.objects.create(
            user=user_obj,
            user_email=user_email,
            user_role=user_role,
            action=action,
            description=description[:500],
            target_type=target_type[:50] if target_type else '',
            target_id=str(target_id)[:50] if target_id else '',
            ip_address=ip,
            user_agent=ua,
        )
    except Exception as e:
        print(f"Audit log error (non-fatal): {e}")

# ============== HELPERS ==============

def check_image_quality(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Could not read image file."
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur_score < 100:
        return False, f"Image is too blurry (Score: {blur_score:.1f}). Please retake in good lighting."
    return True, f"Image is clear (Score: {blur_score:.1f})"

def draw_detections(image_path, sahi_result, output_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0
        count = 0
        for obj in sahi_result.object_prediction_list:
            box = obj.bbox
            x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
            label = f"{obj.category.name} {obj.score.value*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (0, 200, 0), -1)
            cv2.putText(img, label, (x1 + 5, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            count += 1
        cv2.imwrite(output_path, img)
        return True, count
    except Exception as e:
        print(f"draw_detections error: {e}")
        return False, 0

def collect_segmentation_output(seg_project_dir, upload_dir, original_filename):
    try:
        seg_dir = os.path.join(seg_project_dir, 'seg')
        if not os.path.isdir(seg_dir):
            return None
        stem = os.path.splitext(original_filename)[0]
        candidates = [
            os.path.join(seg_dir, stem + '.jpg'),
            os.path.join(seg_dir, stem + '.jpeg'),
            os.path.join(seg_dir, stem + '.png'),
        ]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if src is None:
            imgs = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                src = max(imgs, key=os.path.getmtime)
            else:
                return None
        dest_name = 'seg_' + original_filename
        dest_path = os.path.join(upload_dir, dest_name)
        shutil.copy(src, dest_path)
        return dest_name
    except Exception as e:
        print(f"collect_segmentation_output error: {e}")
        return None

def generate_gradcam_overlay(model_path, image_path, output_path):
    try:
        yolo_model = YOLO(model_path)
        net = yolo_model.model
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return False
        orig_h, orig_w = img_bgr.shape[:2]
        img_640 = cv2.resize(img_bgr, (640, 640))
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        candidate_indices = [6, 5, 4, 7, 8]
        for idx in candidate_indices:
            try:
                layer = net.model[idx]
            except (IndexError, AttributeError, KeyError):
                continue
            activations, gradients = {}, {}

            def fwd_hook(module, inp, out, _s=activations):
                _s['v'] = out[0] if isinstance(out, (tuple, list)) else out

            def bwd_hook(module, gin, gout, _s=gradients):
                _s['v'] = gout[0] if isinstance(gout, (tuple, list)) else gout

            h_f = layer.register_forward_hook(fwd_hook)
            h_b = layer.register_full_backward_hook(bwd_hook)
            try:
                net.eval()
                for p in net.parameters():
                    p.requires_grad = True
                net.zero_grad()
                with torch.enable_grad():
                    preds = net(input_tensor)
                pred_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds
                score = pred_tensor.max()
                score.backward()
                acts, grads = activations.get('v'), gradients.get('v')
                if acts is None or grads is None:
                    continue
                if acts.dim() == 3: acts = acts.unsqueeze(0)
                if grads.dim() == 3: grads = grads.unsqueeze(0)
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=(640, 640), mode='bilinear', align_corners=False)
                cam = cam.squeeze().detach().cpu().numpy()
                cmin, cmax = float(cam.min()), float(cam.max())
                if cmax - cmin < 1e-4:
                    continue
                cam = (cam - cmin) / (cmax - cmin)
                cam_orig = cv2.resize(cam, (orig_w, orig_h))
                heatmap = cv2.applyColorMap((cam_orig * 255).astype('uint8'), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
                cv2.imwrite(output_path, overlay)
                return True
            finally:
                try: h_f.remove()
                except Exception: pass
                try: h_b.remove()
                except Exception: pass
        return False
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return False

def _pdf_make_image(path, max_width_mm=80, max_height_mm=70):
    if not path or not os.path.exists(path):
        return None
    try:
        with PILImage.open(path) as im:
            w, h = im.size
        aspect = (h / w) if w else 1.0
        w_mm = max_width_mm
        h_mm = w_mm * aspect
        if h_mm > max_height_mm:
            h_mm = max_height_mm
            w_mm = h_mm / aspect
        return RLImage(path, width=w_mm*mm, height=h_mm*mm)
    except Exception as e:
        print(f"PDF image error: {e}")
        return None

def generate_pdf_report(scan, patient_name, patient_email):
    upload_dir = os.path.join(BASE_DIR, 'media', 'uploads')
    original_name = os.path.basename(scan.image_path)
    original_path = os.path.join(upload_dir, original_name)
    annotated_path = os.path.join(upload_dir, 'annotated_' + original_name)
    seg_path = os.path.join(upload_dir, 'seg_' + original_name)
    gradcam_path = os.path.join(upload_dir, 'gradcam_' + original_name)

    patient_id_str = '—'
    date_of_birth_str = '—'
    chronic_str = '—'
    allergies_str = '—'
    try:
        if scan.patient:
            profile = scan.patient
            patient_id_str = profile.patient_id or '—'
            date_of_birth_str = profile.date_of_birth.strftime('%B %d, %Y') if profile.date_of_birth else '—'
            chronic_str = (profile.chronic_conditions or '').strip() or '—'
            allergies_str = (profile.allergies or '').strip() or '—'
    except Exception:
        pass

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title="DentIQ Analysis Report"
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=22, textColor=HexColor('#0891B2'), spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=11, textColor=HexColor('#64748B'), spaceAfter=18, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=HexColor('#0F172A'), spaceBefore=10, spaceAfter=8, fontName='Helvetica-Bold')
    h3_style = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11, textColor=HexColor('#0891B2'), spaceBefore=8, spaceAfter=4, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=10, textColor=HexColor('#1E293B'), spaceAfter=6, leading=14)
    small_style = ParagraphStyle('S2', parent=styles['Normal'], fontSize=9, textColor=HexColor('#64748B'), leading=12)
    warn_style = ParagraphStyle('W', parent=styles['Normal'], fontSize=10, textColor=HexColor('#7F1D1D'), leading=14, spaceAfter=4)

    story = []
    story.append(Paragraph("DentIQ", title_style))
    story.append(Paragraph("AI-Powered Dental Diagnosis Report", subtitle_style))

    scan_date = scan.timestamp.strftime("%B %d, %Y at %I:%M %p")
    info_data = [
        ["Patient Name", patient_name or "—"], ["Patient ID", patient_id_str], ["Email", patient_email or "—"],
        ["Date of Birth", date_of_birth_str], ["Report Date", scan_date], ["Report ID", f"DENTIQ-{scan.id:06d}"],
    ]
    info_table = Table(info_data, colWidths=[40*mm, 125*mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#0F172A')),
        ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#475569')), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('LINEBELOW', (0, 0), (-1, -2), 0.3, HexColor('#E2E8F0')),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Clinical Context", h3_style))
    clinical_data = [["Chronic Conditions", chronic_str], ["Known Allergies", allergies_str]]
    clinical_table = Table(clinical_data, colWidths=[40*mm, 125*mm])
    clinical_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10), ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#0F172A')),
        ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#475569')), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('LINEBELOW', (0, 0), (-1, -2), 0.3, HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(clinical_table)
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Diagnosis Summary", h2_style))
    is_healthy = scan.detected_diseases == 'Healthy'
    if is_healthy:
        story.append(Paragraph("<b>No abnormalities detected.</b> The AI model did not identify any of the four supported dental conditions in this image.", body_style))
    else:
        story.append(Paragraph(f"<b>Abnormalities detected:</b> {scan.detected_diseases}", warn_style))
        if scan.confidence_scores and scan.confidence_scores != 'N/A':
            story.append(Paragraph(f"Confidence scores: {scan.confidence_scores}", small_style))
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Image Analysis", h2_style))
    orig_img = _pdf_make_image(original_path)
    anno_img = _pdf_make_image(annotated_path)
    grad_img = _pdf_make_image(gradcam_path)
    seg_img = _pdf_make_image(seg_path)

    row1 = [orig_img or Paragraph("(original not available)", small_style), anno_img or Paragraph("(no detections to annotate)", small_style)]
    t1 = Table([row1], colWidths=[85*mm, 85*mm])
    t1.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('BOTTOMPADDING', (0, 0), (-1, -1), 12)]))
    story.append(t1)

    row2 = [grad_img or Paragraph("(no EigenCAM heatmap)", small_style), seg_img or Paragraph("(no segmentation — caries not detected)", small_style)]
    t2 = Table([row2], colWidths=[85*mm, 85*mm])
    t2.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('BOTTOMPADDING', (0, 0), (-1, -1), 12)]))
    story.append(t2)

    story.append(Paragraph("<b>Top row:</b> original image (left) and bounding-box detections (right). <b>Bottom row:</b> EigenCAM attention heatmap (left) and per-lesion segmentation mask (right, only shown when caries is detected).", small_style))
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("Medical Disclaimer", h2_style))
    story.append(Paragraph("This report is generated by an AI screening tool and is intended for preliminary evaluation only. It does not constitute a medical diagnosis and does not replace examination by a licensed dental professional. All findings should be confirmed through clinical examination and, where indicated, radiographs.", small_style))

    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(f"Generated by <b>DentIQ</b> on {datetime.now().strftime('%B %d, %Y')}", ParagraphStyle('F', parent=small_style, alignment=TA_CENTER)))

    doc.build(story)
    return buffer.getvalue()

def generate_certificate_pdf(cert, patient, doctor):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm, title="DentIQ Medical Certificate")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=22, textColor=HexColor('#0891B2'), spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=11, textColor=HexColor('#64748B'), spaceAfter=18, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, textColor=HexColor('#0F172A'), spaceBefore=12, spaceAfter=6, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=11, textColor=HexColor('#1E293B'), spaceAfter=8, leading=16)
    story = [
        Paragraph("DentIQ Medical Certificate", title_style),
        Paragraph("Issued by DentIQ Dental Clinic", subtitle_style),
    ]
    patient_id_str = "—"
    try:
        patient_id_str = patient.patient_profile.patient_id
    except Exception:
        pass
    cert_info = Table([
        ["Certificate ID", f"CERT-{cert.id:06d}"], ["Issue Date", cert.issued_date.strftime("%B %d, %Y")],
        ["Patient Name", patient.name], ["Patient ID", patient_id_str], ["Email", patient.email],
        ["Doctor", doctor.name if doctor else "—"], ["Specialization", doctor.specialization if doctor else "—"],
    ], colWidths=[45*mm, 120*mm])
    cert_info.setStyle(TableStyle([('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 5), ('LINEBELOW', (0, 0), (-1, -2), 0.3, HexColor('#E2E8F0'))]))
    story.append(cert_info)
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("Diagnosis", h2_style))
    story.append(Paragraph(cert.diagnosis or "—", body_style))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Recommendation", h2_style))
    story.append(Paragraph(cert.recommendation or "—", body_style))
    story.append(Spacer(1, 6*mm))
    if cert.rest_days:
        story.append(Paragraph(f"<b>Rest recommended:</b> {cert.rest_days} day(s)", body_style))
    if cert.valid_until:
        story.append(Paragraph(f"<b>Valid until:</b> {cert.valid_until.strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 20*mm))
    story.append(Paragraph("____________________________", body_style))
    story.append(Paragraph(f"<b>{doctor.name if doctor else 'Authorized Doctor'}</b><br/>{doctor.specialization if doctor else ''}", body_style))
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph(f"Generated by DentIQ on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ParagraphStyle('F', parent=styles['Normal'], fontSize=8, textColor=HexColor('#94A3B8'), alignment=TA_CENTER)))
    doc.build(story)
    return buffer.getvalue()

# ============== AUTH ROUTES ==============

def home(request):
    if 'user_id' in request.session:
        return redirect('dashboard')
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            return redirect('home')
        hashed = generate_password_hash(password)
        user = User.objects.create(name=name, email=email, password=hashed, role='patient')
        PatientProfile.objects.get_or_create(user=user, defaults={'is_verified': False})
        request.session['user_id'] = user.id
        log_action(request, 'register', f"New patient registered: {email}", 'user', user.id)
        request.session.pop('user_id', None)
        messages.success(request, "Registration successful! You can now sign in as a patient.")
        return redirect('home')
    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        selected_role = request.POST.get('role', 'patient')
        try:
            user = User.objects.get(email=email)
            if not user.is_active:
                log_action(request, 'login_failed', f"Deactivated account attempt: {email}")
                messages.error(request, "This account has been deactivated.")
                return redirect('home')
            if check_password_hash(user.password, password):
                if user.role != selected_role:
                    log_action(request, 'login_failed', f"Role mismatch: {email} (chose {selected_role})")
                    messages.error(request, f"This account is registered as '{user.role.upper()}'.")
                    return redirect('home')
                request.session['user_id'] = user.id
                request.session['name'] = user.name
                request.session['role'] = user.role
                request.session['email'] = user.email
                log_action(request, 'login', f"Login successful as {user.role}", 'user', user.id)
                if user.role == 'doctor':
                    return redirect('doctor_dashboard')
                elif user.role == 'admin':
                    return redirect('admin_dashboard')
                else:
                    return redirect('dashboard')
            else:
                log_action(request, 'login_failed', f"Wrong password: {email}")
        except User.DoesNotExist:
            log_action(request, 'login_failed', f"Unknown email: {email}")
        messages.error(request, "Invalid email or password!")
    return redirect('home')

def get_role(request):
    email = request.GET.get('email', '').strip()
    if not email:
        return JsonResponse({'role': None})
    try:
        return JsonResponse({'role': User.objects.get(email=email).role})
    except User.DoesNotExist:
        return JsonResponse({'role': None})

def logout_view(request):
    log_action(request, 'logout', 'User logged out')
    request.session.flush()
    return redirect('home')

# ============== PASSWORD RESET ==============

def forgot_password(request):
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        if not email:
            messages.error(request, "Please enter your email address.")
            return redirect('forgot_password')
        if not User.objects.filter(email=email).exists():
            messages.error(request, "No account found with this email address.")
            return redirect('forgot_password')
        PasswordResetRequest.objects.filter(email=email, used=False).update(used=True)
        reset_req = PasswordResetRequest.objects.create(email=email, code=PasswordResetRequest.generate_code())
        request.session['reset_request_id'] = reset_req.id
        request.session['reset_email'] = email
        log_action(request, 'other', f"Password reset requested for {email}")
        messages.success(request, f"A 6-digit verification code has been generated for {email}.")
        return redirect('verify_reset_code')
    return render(request, 'forgot_password.html')

def verify_reset_code(request):
    reset_req_id = request.session.get('reset_request_id')
    email = request.session.get('reset_email')
    if not reset_req_id or not email:
        messages.error(request, "Please start by entering your email address.")
        return redirect('forgot_password')
    try:
        reset_req = PasswordResetRequest.objects.get(id=reset_req_id)
    except PasswordResetRequest.DoesNotExist:
        messages.error(request, "Reset session expired.")
        return redirect('forgot_password')
    if reset_req.is_expired():
        messages.error(request, "The code has expired.")
        return redirect('forgot_password')
    if request.method == 'POST':
        entered_code = request.POST.get('code', '').strip()
        if not entered_code:
            messages.error(request, "Please enter the 6-digit code.")
            return render(request, 'reset_password.html', {'step': 'verify', 'email': email, 'demo_code': reset_req.code})
        if entered_code != reset_req.code:
            messages.error(request, "Incorrect code.")
            return render(request, 'reset_password.html', {'step': 'verify', 'email': email, 'demo_code': reset_req.code})
        request.session['reset_code_verified'] = True
        return redirect('reset_password')
    return render(request, 'reset_password.html', {'step': 'verify', 'email': email, 'demo_code': reset_req.code})

def reset_password(request):
    reset_req_id = request.session.get('reset_request_id')
    email = request.session.get('reset_email')
    code_verified = request.session.get('reset_code_verified', False)
    if not reset_req_id or not email or not code_verified:
        messages.error(request, "Please verify your code first.")
        return redirect('forgot_password')
    try:
        reset_req = PasswordResetRequest.objects.get(id=reset_req_id)
    except PasswordResetRequest.DoesNotExist:
        messages.error(request, "Reset session expired.")
        return redirect('forgot_password')
    if reset_req.is_expired():
        messages.error(request, "The reset session has expired.")
        return redirect('forgot_password')
    if request.method == 'POST':
        new_password = request.POST.get('new_password', '')
        confirm_password = request.POST.get('confirm_password', '')
        if len(new_password) < 6:
            messages.error(request, "Password must be at least 6 characters.")
            return render(request, 'reset_password.html', {'step': 'new_password', 'email': email})
        if new_password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, 'reset_password.html', {'step': 'new_password', 'email': email})
        try:
            user = User.objects.get(email=email)
            user.password = generate_password_hash(new_password)
            user.save()
            request.session['user_id'] = user.id
            log_action(request, 'password_reset', f"Password reset for {email}", 'user', user.id)
            request.session.pop('user_id', None)
        except User.DoesNotExist:
            messages.error(request, "Account not found.")
            return redirect('home')
        reset_req.used = True
        reset_req.used_at = timezone.now()
        reset_req.save()
        request.session.pop('reset_request_id', None)
        request.session.pop('reset_email', None)
        request.session.pop('reset_code_verified', None)
        messages.success(request, "Password reset successfully!")
        return redirect('home')
    return render(request, 'reset_password.html', {'step': 'new_password', 'email': email})

def resend_reset_code(request):
    email = request.session.get('reset_email')
    if not email:
        messages.error(request, "Please start by entering your email address.")
        return redirect('forgot_password')
    PasswordResetRequest.objects.filter(email=email, used=False).update(used=True)
    new_req = PasswordResetRequest.objects.create(email=email, code=PasswordResetRequest.generate_code())
    request.session['reset_request_id'] = new_req.id
    request.session['reset_code_verified'] = False
    messages.success(request, "A new code has been generated.")
    return redirect('verify_reset_code')

# ============== PATIENT PROFILE ==============

def patient_profile_view(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.session.get('role') != 'patient':
        messages.error(request, "Only patients have a personal profile.")
        return redirect('dashboard')
    try:
        user = User.objects.get(id=request.session['user_id'])
    except User.DoesNotExist:
        return redirect('home')
    profile, _ = PatientProfile.objects.get_or_create(user=user)
    scans = DetectionHistory.objects.filter(user=user).order_by('-timestamp')[:10]
    appointments = Appointment.objects.filter(patient=user).order_by('-appointment_date')[:10]
    visits = Visit.objects.filter(patient=profile).order_by('-visit_date')
    certificates = MedicalCertificate.objects.filter(patient=profile).order_by('-issued_date')
    return render(request, 'patient_profile.html', {
        'user': user, 'profile': profile, 'scans': scans,
        'appointments': appointments, 'visits': visits, 'certificates': certificates,
    })

def edit_patient_profile(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.session.get('role') != 'patient':
        messages.error(request, "Only patients can edit a personal profile.")
        return redirect('dashboard')
    try:
        user = User.objects.get(id=request.session['user_id'])
    except User.DoesNotExist:
        return redirect('home')
    profile, _ = PatientProfile.objects.get_or_create(user=user)
    if request.method == 'POST':
        profile.phone = request.POST.get('phone', '')
        profile.date_of_birth = request.POST.get('date_of_birth') or None
        profile.gender = request.POST.get('gender', '')
        profile.cnic = request.POST.get('cnic', '')
        profile.address = request.POST.get('address', '')
        profile.city = request.POST.get('city', '')
        profile.chronic_conditions = request.POST.get('chronic_conditions', '')
        profile.allergies = request.POST.get('allergies', '')
        profile.current_medications = request.POST.get('current_medications', '')
        profile.past_surgeries = request.POST.get('past_surgeries', '')
        profile.smoker = request.POST.get('smoker') == 'on'
        profile.pregnant = request.POST.get('pregnant') == 'on'
        profile.insurance_provider = request.POST.get('insurance_provider', '')
        profile.insurance_number = request.POST.get('insurance_number', '')
        profile.emergency_contact_name = request.POST.get('emergency_contact_name', '')
        profile.emergency_contact_phone = request.POST.get('emergency_contact_phone', '')
        profile.save()
        messages.success(request, "Profile updated successfully.")
        return redirect('patient_profile')
    return render(request, 'edit_patient_profile.html', {'user': user, 'profile': profile})

def request_verification(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.session.get('role') != 'patient':
        messages.error(request, "Only patients can request verification.")
        return redirect('dashboard')
    try:
        user = User.objects.get(id=request.session['user_id'])
        profile = user.patient_profile
    except (User.DoesNotExist, PatientProfile.DoesNotExist):
        messages.error(request, "Patient profile not found.")
        return redirect('dashboard')
    if profile.is_verified:
        messages.info(request, "Your account is already verified.")
        return redirect('patient_profile')
    if profile.verification_requested:
        messages.info(request, "Your verification request is already pending.")
        return redirect('patient_profile')
    profile.verification_requested = True
    profile.verification_requested_at = timezone.now()
    profile.save()
    log_action(request, 'other', f"Verification requested by patient {profile.patient_id}", 'patient', profile.patient_id)
    messages.success(request, "Verification request submitted.")
    return redirect('patient_profile')

# ============== PATIENT DASHBOARD & SCANS ==============

def dashboard(request):
    if 'user_id' not in request.session:
        return redirect('home')
    role = request.session.get('role')
    if role == 'doctor':
        return redirect('doctor_dashboard')
    if role == 'admin':
        return redirect('admin_dashboard')
    recent_scans = DetectionHistory.objects.filter(user_id=request.session['user_id']).order_by('-timestamp')[:3]
    total_scans = DetectionHistory.objects.filter(user_id=request.session['user_id']).count()
    is_verified = False
    verification_requested = False
    try:
        user_obj = User.objects.get(id=request.session['user_id'])
        if hasattr(user_obj, 'patient_profile'):
            is_verified = user_obj.patient_profile.is_verified
            verification_requested = user_obj.patient_profile.verification_requested
    except Exception:
        pass
    return render(request, 'dashboard.html', {
        'recent_scans': recent_scans, 'total_scans': total_scans,
        'is_verified': is_verified, 'verification_requested': verification_requested,
    })

def scan_history(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.session.get('role') != 'patient':
        return redirect('dashboard')
    scans = DetectionHistory.objects.filter(user_id=request.session['user_id']).order_by('-timestamp')
    return render(request, 'scan_history.html', {'scans': scans})

def analyze(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.method == 'POST' and request.FILES.get('image'):
        file = request.FILES['image']
        upload_dir = os.path.join(BASE_DIR, 'media', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.name)
        with open(filepath, 'wb+') as dest:
            for chunk in file.chunks():
                dest.write(chunk)
        
        # 1. TEETH CHECK (ZERO-SHOT CLIP MODEL)
        try:
            # We initialize the pipeline here (it caches after the first run)
            image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
            candidate_labels = ["a close-up photo of teeth inside a mouth", "a face or selfie", "a keyboard, document, carpet, or random object"]
            
            img = PILImage.open(filepath)
            clip_results = image_classifier(img, candidate_labels=candidate_labels)
            
            # Get the top prediction
            top_prediction = clip_results[0]
            print(f"CLIP Check: {top_prediction['label']} ({top_prediction['score']:.2f})")
            
            # If it's NOT the teeth, reject it immediately
            if top_prediction['label'] != "a close-up photo of teeth inside a mouth" or top_prediction['score'] < 0.60:
                os.remove(filepath)
                return render(request, 'dashboard.html', {'error': "⚠️ Please upload a clear, close-up picture of your teeth. The system detected a face, object, or background instead."})
        except Exception as e:
            print(f"CLIP Error (Skipping): {e}")

        # 2. BLUR CHECK
        is_clear, message = check_image_quality(filepath)
        print(f"Blur check: {message}")
        if not is_clear:
            # If it's a blurry dental image, we remove it and show the blur warning
            os.remove(filepath)
            return render(request, 'dashboard.html', {'error': message})

        # 3. YOLO DISEASE DETECTION
        sahi_result = None
        detected_labels = []
        confidence_scores = []
        try:
            sahi_result = get_sliced_prediction(
                filepath, detection_model,
                slice_height=320, slice_width=320,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_match_threshold=0.5, verbose=False
            )
            for obj in sahi_result.object_prediction_list:
                detected_labels.append(obj.category.name)
                confidence_scores.append(round(obj.score.value, 3))
        except Exception as e:
            print(f"SAHI error: {e}")
            
        annotated_name = None
        if sahi_result is not None and len(detected_labels) > 0:
            annotated_name = 'annotated_' + file.name
            annotated_path = os.path.join(upload_dir, annotated_name)
            ok, _ = draw_detections(filepath, sahi_result, annotated_path)
            if not ok:
                annotated_name = None
                
        segmentation_name = None
        has_caries = any(lbl.lower() == 'caries' for lbl in detected_labels)
        if has_caries:
            try:
                seg_project_dir = os.path.join(BASE_DIR, 'media', 'runs')
                seg_dir = os.path.join(seg_project_dir, 'seg')
                if os.path.isdir(seg_dir):
                    for f in os.listdir(seg_dir):
                        try:
                            os.remove(os.path.join(seg_dir, f))
                        except Exception:
                            pass
                segmentation_model.predict(
                    source=filepath, conf=0.25, save=True,
                    project=seg_project_dir, name='seg',
                    exist_ok=True, verbose=False,
                )
                segmentation_name = collect_segmentation_output(seg_project_dir, upload_dir, file.name)
            except Exception as e:
                print(f"Segmentation error: {e}")
                
        gradcam_name = None
        if sahi_result is not None and len(detected_labels) > 0:
            gradcam_name = 'gradcam_' + file.name
            gradcam_path = os.path.join(upload_dir, gradcam_name)
            ok = generate_gradcam_overlay(
                os.path.join(BASE_DIR, 'models', 'detect_augmented_final.pt'),
                filepath, gradcam_path
            )
            if not ok:
                gradcam_name = None
                
        MERGE_MAP = {'caries': 'Caries / Discoloration', 'tooth_discoloration': 'Caries / Discoloration'}
        merged_labels = [MERGE_MAP.get(lbl, lbl) for lbl in detected_labels]
        label_counts = Counter(merged_labels)
        grouped = []
        for label, count in label_counts.items():
            confs = [confidence_scores[i] for i, l in enumerate(merged_labels) if l == label]
            avg_conf = sum(confs) / len(confs) if confs else 0
            grouped.append({
                'label': label if label == 'Caries / Discoloration' else label.title(),
                'count': count, 'avg_conf': round(avg_conf * 100, 1),
            })
        grouped.sort(key=lambda g: g['count'], reverse=True)
        is_healthy = len(detected_labels) == 0
        history_disease = (", ".join(f"{g['label']} x{g['count']}" for g in grouped) if not is_healthy else "Healthy")
        history_conf = ", ".join(str(c) for c in confidence_scores) if confidence_scores else "N/A"
        user_obj = User.objects.filter(id=request.session['user_id']).first()
        patient_prof = None
        if user_obj:
            patient_prof, _ = PatientProfile.objects.get_or_create(user=user_obj)
        history = DetectionHistory.objects.create(
            user=user_obj, patient=patient_prof, image_path=filepath,
            detected_diseases=history_disease, confidence_scores=history_conf,
            timestamp=datetime.now()
        )
        log_action(request, 'scan_created', f"AI scan performed: {history_disease}", 'scan', history.id)
        return redirect('view_scan', scan_id=history.id)
    return redirect('dashboard')

def serve_upload(request, filename):
    if 'user_id' not in request.session:
        return redirect('home')
    safe_name = os.path.basename(filename)
    path = os.path.join(BASE_DIR, 'media', 'uploads', safe_name)
    if not os.path.exists(path):
        raise Http404("File not found")
    return FileResponse(open(path, 'rb'))

def view_scan(request, scan_id):
    if 'user_id' not in request.session:
        return redirect('home')

    try:
        scan = DetectionHistory.objects.get(id=scan_id)
    except DetectionHistory.DoesNotExist:
        raise Http404("Scan not found")

    role = request.session.get('role')
    if role == 'patient' and scan.user_id != request.session['user_id']:
        raise Http404("Scan not found")

    original_name = os.path.basename(scan.image_path)
    annotated_name = 'annotated_' + original_name if os.path.exists(
        os.path.join(BASE_DIR, 'media', 'uploads', 'annotated_' + original_name)) else None
    seg_name = 'seg_' + original_name if os.path.exists(
        os.path.join(BASE_DIR, 'media', 'uploads', 'seg_' + original_name)) else None
    gradcam_name = 'gradcam_' + original_name if os.path.exists(
        os.path.join(BASE_DIR, 'media', 'uploads', 'gradcam_' + original_name)) else None

    grouped = []
    total_regions = 0
    is_irrelevant = False # <-- NEW FLAG

    disease_str = (scan.detected_diseases or '').strip()
    is_healthy = (disease_str == '' or disease_str.lower() == 'healthy')

    if not is_healthy:
        raw_confs = []
        if scan.confidence_scores and scan.confidence_scores != 'N/A':
            for token in scan.confidence_scores.split(','):
                token = token.strip()
                try:
                    raw_confs.append(float(token))
                except (ValueError, TypeError):
                    pass

        conds = []
        for part in [x.strip() for x in disease_str.split(',') if x.strip()]:
            m = re.match(r'^(.*?)\s*[xX]\s*(\d+)\s*$', part)
            if m:
                label = m.group(1).strip()
                try:
                    cnt = int(m.group(2))
                except ValueError:
                    cnt = 1
                conds.append((label, cnt))
                total_regions += cnt
            else:
                conds.append((part, 1))
                total_regions += 1

        if conds:
            total_detections = sum(c for _, c in conds) or 1
            if raw_confs and len(raw_confs) == total_detections:
                idx = 0
                for label, cnt in conds:
                    slice_confs = raw_confs[idx:idx + cnt]
                    idx += cnt
                    avg = sum(slice_confs) / len(slice_confs) if slice_confs else 0
                    grouped.append({'label': label, 'count': cnt, 'avg_conf': round(avg * 100, 1)})
            elif raw_confs:
                overall = sum(raw_confs) / len(raw_confs)
                for label, cnt in conds:
                    grouped.append({'label': label, 'count': cnt, 'avg_conf': round(overall * 100, 1)})
            else:
                for label, cnt in conds:
                    grouped.append({'label': label, 'count': cnt, 'avg_conf': None})

    for g in grouped:
        if g['avg_conf'] is None:
            g['severity'] = 'unknown'; g['color'] = '#64748B'; g['bg_color'] = '#F1F5F9'; g['severity_label'] = 'Unavailable'
        elif g['avg_conf'] >= 75:
            g['severity'] = 'high'; g['color'] = '#059669'; g['bg_color'] = '#ECFDF5'; g['severity_label'] = 'High reliability'
        elif g['avg_conf'] >= 50:
            g['severity'] = 'moderate'; g['color'] = '#D97706'; g['bg_color'] = '#FEF3C7'; g['severity_label'] = 'Moderate'
        else:
            g['severity'] = 'low'; g['color'] = '#DC2626'; g['bg_color'] = '#FEF2F2'; g['severity_label'] = 'Low — interpret with caution'

    # NEW LOGIC: If the confidence is extremely low (< 45%) or total regions > 10 (detecting everything), it's likely a false positive on a non-dental image
    if grouped and (total_regions > 10 or all(g['avg_conf'] and g['avg_conf'] < 45 for g in grouped)):
        is_irrelevant = True

    return render(request, 'results.html', {
        'scan_id': scan.id, 'original': original_name,
        'annotated': annotated_name, 'segmentation': seg_name, 'gradcam': gradcam_name,
        'detected': grouped, 'total_regions': total_regions,
        'is_healthy': is_healthy,
        'is_irrelevant': is_irrelevant, # <-- PASSED TO TEMPLATE
    })

def download_report(request, scan_id):
    if 'user_id' not in request.session:
        return redirect('home')
    try:
        scan = DetectionHistory.objects.get(id=scan_id)
    except DetectionHistory.DoesNotExist:
        raise Http404("Scan not found")
    role = request.session.get('role')
    if role == 'patient' and scan.user_id != request.session['user_id']:
        raise Http404("Scan not found")
    try:
        pdf_bytes = generate_pdf_report(scan,
            patient_name=request.session.get('name', ''),
            patient_email=request.session.get('email', ''))
    except Exception as e:
        print(f"PDF generation error: {e}")
        messages.error(request, "Could not generate PDF report.")
        return redirect('view_scan', scan_id=scan.id)
    log_action(request, 'report_downloaded', f"Report downloaded for scan #{scan.id}", 'scan', scan.id)
    filename = f"DentIQ_Report_{scan.id:06d}.pdf"
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

# ============== SETTINGS ==============

def settings_view(request):
    if 'user_id' not in request.session:
        return redirect('home')
    try:
        user = User.objects.get(id=request.session['user_id'])
    except User.DoesNotExist:
        return redirect('home')
    return render(request, 'settings.html', {'user': user})

def change_password(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.method == 'POST':
        current = request.POST.get('current_password', '')
        new = request.POST.get('new_password', '')
        confirm = request.POST.get('confirm_password', '')
        try:
            user = User.objects.get(id=request.session['user_id'])
        except User.DoesNotExist:
            return redirect('home')
        if not check_password_hash(user.password, current):
            messages.error(request, "Current password is incorrect.")
            return redirect('settings')
        if len(new) < 6:
            messages.error(request, "New password must be at least 6 characters.")
            return redirect('settings')
        if new != confirm:
            messages.error(request, "New passwords do not match.")
            return redirect('settings')
        user.password = generate_password_hash(new)
        user.save()
        log_action(request, 'other', 'Password changed from settings page', 'user', user.id)
        messages.success(request, "Password updated successfully.")
    return redirect('settings')

# ============== APPOINTMENTS ==============

def appointments(request):
    if 'user_id' not in request.session:
        return redirect('home')
    role = request.session.get('role')
    if role == 'admin':
        messages.info(request, "Admins don't have personal appointments.")
        return redirect('admin_dashboard')
    if role == 'patient':
        appts_qs = Appointment.objects.filter(patient_id=request.session['user_id']).select_related('doctor').order_by('-appointment_date', '-appointment_time')
        appts = []
        for a in appts_qs:
            existing = Review.objects.filter(user_id=request.session['user_id'], doctor=a.doctor).first()
            appts.append({'obj': a, 'reviewable': (a.status == 'completed' and not existing), 'review': existing})
        return render(request, 'appointments.html', {'patient_appointments': appts, 'role': 'patient'})
    try:
        doctor = Doctor.objects.get(email=request.session.get('email'))
        appts = Appointment.objects.filter(doctor=doctor).select_related('patient').order_by('-appointment_date', '-appointment_time')
        avg_rating = Review.objects.filter(doctor=doctor).aggregate(Avg('rating'))['rating__avg'] or 0
        review_count = Review.objects.filter(doctor=doctor).count()
        recent_reviews = Review.objects.filter(doctor=doctor).select_related('user').order_by('-created_at')[:5]
    except Doctor.DoesNotExist:
        appts = []; avg_rating = 0; review_count = 0; recent_reviews = []
    return render(request, 'appointments.html', {
        'appointments': appts, 'role': 'doctor',
        'avg_rating': round(avg_rating, 1) if avg_rating else 0,
        'review_count': review_count, 'recent_reviews': recent_reviews,
    })

def book_appointment(request):
    if 'user_id' not in request.session:
        return redirect('home')
    if request.session.get('role') != 'patient':
        messages.error(request, "Only patients can book appointments.")
        return redirect('dashboard')
    try:
        user_obj = User.objects.get(id=request.session['user_id'])
        profile = user_obj.patient_profile
    except (User.DoesNotExist, PatientProfile.DoesNotExist):
        messages.error(request, "Patient profile not found.")
        return redirect('dashboard')
    if not profile.is_verified:
        messages.warning(request, "Account not yet verified.")
        return redirect('patient_profile')
    latest_scan = DetectionHistory.objects.filter(user=user_obj).order_by('-timestamp').first()
    latest_scan_healthy = (latest_scan and latest_scan.detected_diseases == 'Healthy')
    if request.method == 'POST':
        appt = Appointment.objects.create(
            patient_id=request.session['user_id'],
            doctor_id=request.POST.get('doctor_id'),
            appointment_date=request.POST.get('date'),
            appointment_time=request.POST.get('time') or None,
            reason=request.POST.get('reason', ''), status='pending'
        )
        log_action(request, 'appointment_booked', f"Booked with Dr. {appt.doctor.name} on {appt.appointment_date}", 'appointment', appt.id)
        messages.success(request, "Appointment booked successfully!")
        return redirect('appointments')
    doctors_qs = Doctor.objects.filter(is_active=True)
    doctors_with_ratings = []
    for d in doctors_qs:
        avg = Review.objects.filter(doctor=d).aggregate(Avg('rating'))['rating__avg'] or 0
        cnt = Review.objects.filter(doctor=d).count()
        doctors_with_ratings.append({'obj': d, 'avg_rating': round(avg, 1) if avg else 0, 'review_count': cnt})
    return render(request, 'book_appointment.html', {
        'doctors': doctors_qs, 'doctors_with_ratings': doctors_with_ratings,
        'latest_scan_healthy': latest_scan_healthy, 'latest_scan': latest_scan,
    })

def update_appointment_status(request, appt_id):
    if 'user_id' not in request.session or request.session.get('role') != 'doctor':
        return redirect('home')
    if request.method != 'POST':
        return redirect('appointments')
    try:
        doctor = Doctor.objects.get(email=request.session.get('email'))
    except Doctor.DoesNotExist:
        return redirect('appointments')
    try:
        appt = Appointment.objects.get(id=appt_id, doctor=doctor)
    except Appointment.DoesNotExist:
        return redirect('appointments')
    old_status = appt.status
    new_status = request.POST.get('status', '').strip()
    notes = request.POST.get('doctor_notes', '').strip()
    if new_status not in ('pending', 'confirmed', 'completed', 'cancelled'):
        messages.error(request, "Invalid status.")
        return redirect('appointments')
    appt.status = new_status
    if notes:
        appt.doctor_notes = notes
    appt.save()
    log_action(request, 'appointment_status_changed',
               f"Appointment #{appt.id} ({appt.patient.name}) {old_status} → {new_status}",
               'appointment', appt.id)
    messages.success(request, f"Appointment for {appt.patient.name} marked as '{new_status}'.")
    return redirect('appointments')

def cancel_appointment_patient(request, appt_id):
    if 'user_id' not in request.session or request.session.get('role') != 'patient':
        return redirect('home')
    if request.method != 'POST':
        return redirect('appointments')
    try:
        appt = Appointment.objects.get(id=appt_id, patient_id=request.session['user_id'])
    except Appointment.DoesNotExist:
        messages.error(request, "Appointment not found.")
        return redirect('appointments')
    if appt.status != 'pending':
        messages.warning(request, "Only pending appointments can be cancelled.")
        return redirect('appointments')
    appt.status = 'cancelled'
    appt.save()
    log_action(request, 'appointment_cancelled', f"Patient cancelled appointment #{appt.id}", 'appointment', appt.id)
    messages.success(request, "Appointment cancelled.")
    return redirect('appointments')

# ============== REVIEWS ==============

def leave_review(request, appt_id):
    if 'user_id' not in request.session or request.session.get('role') != 'patient':
        return redirect('home')
    try:
        appt = Appointment.objects.get(id=appt_id, patient_id=request.session['user_id'])
    except Appointment.DoesNotExist:
        return redirect('appointments')
    if appt.status != 'completed':
        messages.warning(request, "You can only review completed appointments.")
        return redirect('appointments')
    existing = Review.objects.filter(user_id=request.session['user_id'], doctor=appt.doctor).first()
    if existing:
        messages.info(request, "You have already reviewed this doctor.")
        return redirect('appointments')
    if request.method == 'POST':
        try:
            rating = int(request.POST.get('rating', '0'))
        except ValueError:
            rating = 0
        if rating < 1 or rating > 5:
            messages.error(request, "Please choose a rating between 1 and 5 stars.")
            return redirect('leave_review', appt_id=appt_id)
        comment = request.POST.get('comment', '').strip()
        review = Review.objects.create(user_id=request.session['user_id'], doctor=appt.doctor, rating=rating, comment=comment)
        log_action(request, 'review_submitted', f"Rated Dr. {appt.doctor.name} {rating}/5", 'review', review.id)
        messages.success(request, f"Thank you! Your review for Dr. {appt.doctor.name} has been submitted.")
        return redirect('appointments')
    return render(request, 'leave_review.html', {'appt': appt, 'doctor': appt.doctor})

def delete_review(request, review_id):
    if 'user_id' not in request.session or request.session.get('role') != 'admin':
        return redirect('home')
    if request.method != 'POST':
        return redirect('admin_dashboard')
    try:
        review = Review.objects.get(id=review_id)
    except Review.DoesNotExist:
        messages.error(request, "Review not found.")
        return redirect('admin_dashboard')
    info = f"Review #{review.id} by {review.user.name} for Dr. {review.doctor.name if review.doctor else 'N/A'}"
    review.delete()
    log_action(request, 'review_deleted', info, 'review', review_id)
    messages.success(request, "Review deleted.")
    return redirect('admin_dashboard')

# ============== DOCTOR ROUTES ==============

def doctor_dashboard(request):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    try:
        doctor = Doctor.objects.get(email=request.session.get('email'))
        appts = Appointment.objects.filter(doctor=doctor).select_related('patient')
    except Doctor.DoesNotExist:
        doctor = None; appts = []
    total = len(appts)
    pending = sum(1 for a in appts if a.status == 'pending')
    confirmed = sum(1 for a in appts if a.status == 'confirmed')
    patient_ids = Appointment.objects.filter(doctor=doctor).values_list('patient_id', flat=True).distinct()
    patient_count = len(set(patient_ids)) if doctor else 0
    scan_count = DetectionHistory.objects.filter(user_id__in=patient_ids).count() if patient_ids else 0
    unverified_count = PatientProfile.objects.filter(
        is_verified=False, verification_requested=True, user__role='patient',
    ).annotate(scans_count=Count('user__detections')).filter(scans_count__gt=0).count()
    pending_appt_count = pending
    if doctor:
        avg_rating = Review.objects.filter(doctor=doctor).aggregate(Avg('rating'))['rating__avg'] or 0
        review_count = Review.objects.filter(doctor=doctor).count()
        recent_reviews = Review.objects.filter(doctor=doctor).select_related('user').order_by('-created_at')[:3]
    else:
        avg_rating = 0; review_count = 0; recent_reviews = []
    return render(request, 'doctor_dashboard.html', {
        'appointments': appts, 'pending_count': pending, 'confirmed_count': confirmed,
        'total_count': total, 'patient_count': patient_count, 'scan_count': scan_count,
        'unverified_count': unverified_count, 'pending_appt_count': pending_appt_count,
        'avg_rating': round(avg_rating, 1) if avg_rating else 0,
        'review_count': review_count, 'recent_reviews': recent_reviews,
    })

def doctor_patient_search(request):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    q = request.GET.get('q', '').strip()
    matches = []
    if q:
        profiles = PatientProfile.objects.filter(
            Q(patient_id__icontains=q) | Q(phone__icontains=q), user__role='patient',
        ).select_related('user')[:20]
        for p in profiles:
            matches.append({'profile': p, 'user': p.user, 'scan_count': DetectionHistory.objects.filter(user=p.user).count()})
    return render(request, 'doctor_patient_search.html', {'q': q, 'matches': matches})

def verify_patients(request):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    unverified = PatientProfile.objects.filter(
        is_verified=False, verification_requested=True, user__role='patient',
    ).annotate(scans_count=Count('user__detections')).filter(scans_count__gt=0).select_related('user').order_by('-verification_requested_at')
    verified = PatientProfile.objects.filter(is_verified=True, user__role='patient').select_related('user').order_by('-verified_at')[:20]
    waitlist = PatientProfile.objects.filter(
        is_verified=False, verification_requested=True, user__role='patient',
    ).annotate(scans_count=Count('user__detections')).filter(scans_count=0).select_related('user').order_by('-verification_requested_at')
    return render(request, 'verify_patient.html', {'unverified': unverified, 'verified': verified, 'waitlist': waitlist})

def verify_patient_action(request, patient_id):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    if request.method != 'POST':
        return redirect('verify_patients')
    try:
        profile = PatientProfile.objects.get(patient_id=patient_id, user__role='patient')
    except PatientProfile.DoesNotExist:
        raise Http404("Patient not found")
    profile.is_verified = True
    profile.verified_at = timezone.now()
    profile.verified_by = request.session.get('email', 'doctor')
    profile.save()
    log_action(request, 'patient_verified', f"Verified patient {patient_id} ({profile.user.name})", 'patient', patient_id)
    messages.success(request, f"Patient {patient_id} has been verified.")
    return redirect('verify_patients')

def unverify_patient_action(request, patient_id):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    if request.method != 'POST':
        return redirect('verify_patients')
    try:
        profile = PatientProfile.objects.get(patient_id=patient_id, user__role='patient')
    except PatientProfile.DoesNotExist:
        raise Http404("Patient not found")
    profile.is_verified = False
    profile.verified_at = None
    profile.verified_by = ''
    profile.save()
    log_action(request, 'patient_unverified', f"Revoked verification for {patient_id}", 'patient', patient_id)
    messages.warning(request, f"Patient {patient_id} verification revoked.")
    return redirect('verify_patients')

def patient_file(request, patient_id):
    if 'user_id' not in request.session or request.session['role'] not in ('doctor', 'admin'):
        return redirect('home')
    try:
        profile = PatientProfile.objects.get(patient_id=patient_id)
    except PatientProfile.DoesNotExist:
        raise Http404("Patient not found")
    patient_user = profile.user
    scans = DetectionHistory.objects.filter(user=patient_user).order_by('-timestamp')
    appointments = Appointment.objects.filter(patient=patient_user).order_by('-appointment_date')
    visits = Visit.objects.filter(patient=profile).order_by('-visit_date')
    documents = PatientDocument.objects.filter(patient=profile).order_by('-uploaded_at')
    certificates = MedicalCertificate.objects.filter(patient=profile).order_by('-issued_date')
    return render(request, 'patient_file.html', {
        'profile': profile, 'patient_user': patient_user,
        'scans': scans, 'appointments': appointments, 'visits': visits,
        'documents': documents, 'certificates': certificates,
    })

def add_visit(request, patient_id):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    try:
        profile = PatientProfile.objects.get(patient_id=patient_id)
    except PatientProfile.DoesNotExist:
        raise Http404("Patient not found")
    try:
        doctor = Doctor.objects.get(email=request.session.get('email'))
    except Doctor.DoesNotExist:
        doctor = None
    if request.method == 'POST':
        follow_up = request.POST.get('follow_up_date') or None
        visit = Visit.objects.create(
            patient=profile, doctor=doctor,
            reason=request.POST.get('reason', ''),
            chief_complaint=request.POST.get('chief_complaint', ''),
            examination_findings=request.POST.get('examination_findings', ''),
            diagnosis=request.POST.get('diagnosis', ''),
            treatment_plan=request.POST.get('treatment_plan', ''),
            notes=request.POST.get('notes', ''),
            follow_up_date=follow_up,
        )
        log_action(request, 'visit_added', f"Visit added for {patient_id}", 'visit', visit.id)
        messages.success(request, "Visit recorded successfully.")
        return redirect('patient_file', patient_id=patient_id)
    return render(request, 'add_visit.html', {'profile': profile})

def add_prescription(request, visit_id):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    visit = get_object_or_404(Visit, id=visit_id)
    if request.method == 'POST':
        presc = Prescription.objects.create(
            visit=visit,
            medication=request.POST.get('medication', ''),
            dose=request.POST.get('dose', ''),
            frequency=request.POST.get('frequency', ''),
            duration=request.POST.get('duration', ''),
            instructions=request.POST.get('instructions', ''),
        )
        log_action(request, 'prescription_added', f"Prescription {presc.medication} added for {visit.patient.patient_id}", 'prescription', presc.id)
        messages.success(request, "Prescription added.")
        return redirect('patient_file', patient_id=visit.patient.patient_id)
    return render(request, 'add_prescription.html', {'visit': visit, 'profile': visit.patient})

def upload_patient_document(request, patient_id):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    try:
        profile = PatientProfile.objects.get(patient_id=patient_id)
    except PatientProfile.DoesNotExist:
        raise Http404("Patient not found")
    if request.method == 'POST' and request.FILES.get('file'):
        doc = PatientDocument.objects.create(
            patient=profile,
            doc_type=request.POST.get('doc_type', 'other'),
            title=request.POST.get('title', 'Untitled'),
            file=request.FILES['file'],
            uploaded_by=request.session.get('email', ''),
            notes=request.POST.get('notes', ''),
        )
        log_action(request, 'document_uploaded', f"Uploaded '{doc.title}' for {patient_id}", 'document', doc.id)
        messages.success(request, "Document uploaded.")
        return redirect('patient_file', patient_id=patient_id)
    return render(request, 'upload_patient_document.html', {'profile': profile})

def issue_certificate(request, patient_id):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    try:
        profile = PatientProfile.objects.get(patient_id=patient_id)
    except PatientProfile.DoesNotExist:
        raise Http404("Patient not found")
    try:
        doctor = Doctor.objects.get(email=request.session.get('email'))
    except Doctor.DoesNotExist:
        doctor = None
    if request.method == 'POST':
        valid_until = request.POST.get('valid_until') or None
        try:
            rest_days = int(request.POST.get('rest_days', '0') or '0')
        except ValueError:
            rest_days = 0
        cert = MedicalCertificate.objects.create(
            patient=profile, doctor=doctor,
            diagnosis=request.POST.get('diagnosis', ''),
            recommendation=request.POST.get('recommendation', ''),
            rest_days=rest_days, valid_until=valid_until,
        )
        log_action(request, 'certificate_issued', f"Certificate CERT-{cert.id:06d} issued for {patient_id}", 'certificate', cert.id)
        messages.success(request, f"Certificate CERT-{cert.id:06d} issued.")
        return redirect('patient_file', patient_id=patient_id)
    return render(request, 'issue_certificate.html', {'profile': profile})

def download_certificate(request, cert_id):
    if 'user_id' not in request.session:
        return redirect('home')
    cert = get_object_or_404(MedicalCertificate, id=cert_id)
    patient = cert.patient.user
    doctor = cert.doctor
    try:
        pdf_bytes = generate_certificate_pdf(cert, patient, doctor)
    except Exception as e:
        print(f"Certificate PDF error: {e}")
        messages.error(request, "Could not generate certificate PDF.")
        return redirect('patient_file', patient_id=cert.patient.patient_id)
    log_action(request, 'certificate_downloaded', f"Certificate CERT-{cert.id:06d} downloaded", 'certificate', cert.id)
    filename = f"DentIQ_Certificate_{cert.id:06d}.pdf"
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

def doctor_patients(request):
    if 'user_id' not in request.session or request.session['role'] != 'doctor':
        return redirect('home')
    try:
        doctor = Doctor.objects.get(email=request.session.get('email'))
    except Doctor.DoesNotExist:
        return render(request, 'doctor_patients.html', {'patients': []})
    patient_ids = Appointment.objects.filter(doctor=doctor).values_list('patient_id', flat=True).distinct()
    patients = User.objects.filter(id__in=patient_ids)
    rows = []
    for p in patients:
        try:
            prof = p.patient_profile
        except PatientProfile.DoesNotExist:
            prof, _ = PatientProfile.objects.get_or_create(user=p)
        scans = DetectionHistory.objects.filter(user=p).order_by('-timestamp')[:5]
        rows.append({
            'patient': p, 'profile': prof,
            'scan_count': DetectionHistory.objects.filter(user=p).count(),
            'recent_scans': scans,
        })
    return render(request, 'doctor_patients.html', {'patients': rows})

# ============== ADMIN ROUTES ==============

def admin_dashboard(request):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    return render(request, 'admin_dashboard.html', {
        'user_count': User.objects.count(),
        'doctor_count': Doctor.objects.count(),
        'appt_count': Appointment.objects.count(),
        'patient_count': PatientProfile.objects.count(),
        'scan_count': DetectionHistory.objects.count(),
        'visit_count': Visit.objects.count(),
        'document_count': PatientDocument.objects.count(),
        'certificate_count': MedicalCertificate.objects.count(),
        'review_count': Review.objects.count(),
        'audit_count': AuditLog.objects.count(),
    })

def admin_scans(request):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    q = request.GET.get('q', '').strip()
    scans_qs = DetectionHistory.objects.all().order_by('-timestamp')
    if q:
        scans_qs = scans_qs.filter(
            Q(detected_diseases__icontains=q) |
            Q(user__email__icontains=q) |
            Q(user__name__icontains=q)
        )
    rows = []
    for scan in scans_qs[:200]:
        rows.append({
            'scan': scan,
            'user_name': scan.user.name if scan.user else '—',
            'user_email': scan.user.email if scan.user else '—',
        })
    return render(request, 'admin_scans.html', {'rows': rows, 'q': q})

def admin_users(request):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    role_filter = request.GET.get('role', '').strip()
    q = request.GET.get('q', '').strip()
    users_qs = User.objects.all().order_by('-id')
    if role_filter in ('patient', 'doctor', 'admin'):
        users_qs = users_qs.filter(role=role_filter)
    if q:
        users_qs = users_qs.filter(Q(name__icontains=q) | Q(email__icontains=q))
    rows = []
    for u in users_qs[:300]:
        extra = {}
        if u.role == 'patient':
            try:
                prof = u.patient_profile
                extra = {
                    'patient_id': prof.patient_id,
                    'is_verified': prof.is_verified,
                    'verification_requested': prof.verification_requested,
                    'scan_count': DetectionHistory.objects.filter(user=u).count(),
                }
            except PatientProfile.DoesNotExist:
                extra = {'patient_id': '—', 'is_verified': False, 'verification_requested': False, 'scan_count': 0}
        rows.append({'user': u, 'extra': extra})
    role_counts = {
        'patient': User.objects.filter(role='patient').count(),
        'doctor': User.objects.filter(role='doctor').count(),
        'admin': User.objects.filter(role='admin').count(),
    }
    return render(request, 'admin_users.html', {
        'rows': rows, 'role_filter': role_filter, 'q': q, 'role_counts': role_counts,
    })

def admin_user_toggle_active(request, user_id):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    if request.method != 'POST':
        return redirect('admin_users')
    try:
        target = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('admin_users')
    if target.id == request.session['user_id']:
        messages.warning(request, "You cannot deactivate your own account.")
        return redirect('admin_users')
    target.is_active = not target.is_active
    target.save()
    action = 'user_reactivated' if target.is_active else 'user_deactivated'
    log_action(request, action, f"{target.name} ({target.email})", 'user', target.id)
    state = "reactivated" if target.is_active else "deactivated"
    messages.success(request, f"{target.name} has been {state}.")
    return redirect('admin_users')

def admin_user_change_role(request, user_id):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    if request.method != 'POST':
        return redirect('admin_users')
    try:
        target = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('admin_users')
    if target.id == request.session['user_id']:
        messages.warning(request, "You cannot change your own role.")
        return redirect('admin_users')
    new_role = request.POST.get('new_role', '').strip()
    if new_role not in ('patient', 'doctor', 'admin'):
        messages.error(request, "Invalid role.")
        return redirect('admin_users')
    old_role = target.role
    target.role = new_role
    target.save()
    if new_role == 'patient':
        PatientProfile.objects.get_or_create(user=target, defaults={'is_verified': False})
    log_action(request, 'user_role_changed', f"{target.name}: {old_role} → {new_role}", 'user', target.id)
    messages.success(request, f"{target.name}'s role changed from '{old_role}' to '{new_role}'.")
    return redirect('admin_users')

def admin_doctors(request):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    q = request.GET.get('q', '').strip()
    doctors_qs = Doctor.objects.all().order_by('-id')
    if q:
        doctors_qs = doctors_qs.filter(
            Q(name__icontains=q) | Q(email__icontains=q) | Q(specialization__icontains=q)
        )
    rows = []
    for d in doctors_qs[:200]:
        login_exists = User.objects.filter(email=d.email, role='doctor').exists()
        appt_count = Appointment.objects.filter(doctor=d).count()
        avg = Review.objects.filter(doctor=d).aggregate(Avg('rating'))['rating__avg'] or 0
        review_count = Review.objects.filter(doctor=d).count()
        rows.append({
            'doctor': d, 'login_exists': login_exists, 'appt_count': appt_count,
            'avg_rating': round(avg, 1) if avg else 0, 'review_count': review_count,
        })
    return render(request, 'admin_doctors.html', {
        'rows': rows, 'q': q,
        'total_doctors': Doctor.objects.count(),
        'active_doctors': Doctor.objects.filter(is_active=True).count(),
    })

def admin_doctor_add(request):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        specialization = request.POST.get('specialization', '').strip()
        available_days = request.POST.get('available_days', '').strip()
        available_times = request.POST.get('available_times', '').strip()
        password = request.POST.get('password', '').strip()
        if not name or not email or not password:
            messages.error(request, "Name, email and password are required.")
            return redirect('admin_doctors')
        if Doctor.objects.filter(email=email).exists():
            messages.error(request, f"A doctor with email {email} already exists.")
            return redirect('admin_doctors')
        if User.objects.filter(email=email).exists():
            messages.error(request, f"A user with email {email} already exists.")
            return redirect('admin_doctors')
        if len(password) < 6:
            messages.error(request, "Password must be at least 6 characters.")
            return redirect('admin_doctors')
        d = Doctor.objects.create(
            name=name, email=email, specialization=specialization,
            available_days=available_days, available_times=available_times, is_active=True,
        )
        User.objects.create(
            name=name, email=email,
            password=generate_password_hash(password), role='doctor', is_active=True,
        )
        log_action(request, 'doctor_added', f"New doctor: {name} ({email})", 'doctor', d.id)
        messages.success(request, f"Doctor {name} has been added successfully.")
        return redirect('admin_doctors')
    return redirect('admin_doctors')

def admin_doctor_edit(request, doctor_id):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    try:
        doctor = Doctor.objects.get(id=doctor_id)
    except Doctor.DoesNotExist:
        raise Http404("Doctor not found")
    if request.method == 'POST':
        doctor.name = request.POST.get('name', doctor.name).strip()
        doctor.specialization = request.POST.get('specialization', doctor.specialization).strip()
        doctor.available_days = request.POST.get('available_days', doctor.available_days).strip()
        doctor.available_times = request.POST.get('available_times', doctor.available_times).strip()
        doctor.save()
        try:
            user = User.objects.get(email=doctor.email, role='doctor')
            user.name = doctor.name
            user.save()
        except User.DoesNotExist:
            pass
        log_action(request, 'doctor_edited', f"Edited Dr. {doctor.name}", 'doctor', doctor.id)
        messages.success(request, f"Doctor {doctor.name} updated.")
        return redirect('admin_doctors')
    return redirect('admin_doctors')

def admin_doctor_toggle_active(request, doctor_id):
    if 'user_id' not in request.session or request.session['role'] != 'admin':
        return redirect('home')
    if request.method != 'POST':
        return redirect('admin_doctors')
    try:
        doctor = Doctor.objects.get(id=doctor_id)
    except Doctor.DoesNotExist:
        messages.error(request, "Doctor not found.")
        return redirect('admin_doctors')
    doctor.is_active = not doctor.is_active
    doctor.save()
    try:
        user = User.objects.get(email=doctor.email, role='doctor')
        user.is_active = doctor.is_active
        user.save()
    except User.DoesNotExist:
        pass
    action = 'doctor_reactivated' if doctor.is_active else 'doctor_deactivated'
    log_action(request, action, f"Dr. {doctor.name}", 'doctor', doctor.id)
    state = "reactivated" if doctor.is_active else "deactivated"
    messages.success(request, f"Dr. {doctor.name} has been {state}.")
    return redirect('admin_doctors')

# ============== MODEL MONITORING ==============

def admin_model_monitoring(request):
    if 'user_id' not in request.session or request.session.get('role') != 'admin':
        return redirect('home')

    detection_model_path = os.path.join(BASE_DIR, 'models', 'detect_augmented_final.pt')
    seg_model_path = os.path.join(BASE_DIR, 'models', 'seg_augmented_final.pt')

    def file_size_mb(path):
        try:
            return round(os.path.getsize(path) / (1024 * 1024), 2)
        except Exception:
            return None

    model_info = {
        'name': 'DentIQ Detection Model',
        'version': 'v1.0 (YOLOv8n — Augmented)',
        'task': 'Object Detection (4-class)',
        'classes': ['calculus', 'caries', 'hypodontia', 'tooth_discoloration'],
        'confidence_threshold': CONF_THRESHOLD,
        'detection_size_mb': file_size_mb(detection_model_path),
        'segmentation_size_mb': file_size_mb(seg_model_path),
        'segmentation_task': 'Instance Segmentation (caries only)',
        'inference_device': 'CPU',
    }

    total_scans = DetectionHistory.objects.count()
    total_users_with_scans = DetectionHistory.objects.values('user').distinct().count()
    healthy_count = DetectionHistory.objects.filter(detected_diseases='Healthy').count()
    abnormal_count = total_scans - healthy_count

    disease_counts = {
        'Caries / Discoloration': 0,
        'Calculus': 0,
        'Hypodontia': 0,
        'Tooth Discoloration': 0,
        'Healthy': healthy_count,
    }
    for scan in DetectionHistory.objects.all():
        label_str = scan.detected_diseases or ''
        if 'Caries' in label_str:
            disease_counts['Caries / Discoloration'] += 1
        if 'Calculus' in label_str:
            disease_counts['Calculus'] += 1
        if 'Hypodontia' in label_str:
            disease_counts['Hypodontia'] += 1
        if 'Tooth Discoloration' in label_str:
            disease_counts['Tooth Discoloration'] += 1

    disease_distribution = []
    for label, count in disease_counts.items():
        pct = round((count / total_scans * 100), 1) if total_scans else 0
        disease_distribution.append({'label': label, 'count': count, 'pct': pct})

    all_confidences = []
    for scan in DetectionHistory.objects.exclude(confidence_scores='').exclude(confidence_scores='N/A'):
        for token in (scan.confidence_scores or '').split(','):
            token = token.strip()
            try:
                val = float(token)
                if 0 <= val <= 1:
                    all_confidences.append(val)
            except (ValueError, TypeError):
                pass

    if all_confidences:
        avg_confidence = round(sum(all_confidences) / len(all_confidences) * 100, 1)
        min_confidence = round(min(all_confidences) * 100, 1)
        max_confidence = round(max(all_confidences) * 100, 1)
        high_conf = sum(1 for c in all_confidences if c >= 0.75)
        mid_conf = sum(1 for c in all_confidences if 0.5 <= c < 0.75)
        low_conf = sum(1 for c in all_confidences if c < 0.5)
    else:
        avg_confidence = min_confidence = max_confidence = 0
        high_conf = mid_conf = low_conf = 0

    confidence_stats = {
        'avg': avg_confidence, 'min': min_confidence, 'max': max_confidence,
        'total_detections': len(all_confidences),
        'high': high_conf, 'moderate': mid_conf, 'low': low_conf,
    }

    now = timezone.now()
    scans_last_7_days = DetectionHistory.objects.filter(timestamp__gte=now - timedelta(days=7)).count()
    scans_last_30_days = DetectionHistory.objects.filter(timestamp__gte=now - timedelta(days=30)).count()

    top_patients = []
    top_users_qs = User.objects.filter(role='patient').annotate(scans=Count('detections')).filter(scans__gt=0).order_by('-scans')[:5]
    for u in top_users_qs:
        top_patients.append({'name': u.name, 'email': u.email, 'scans': u.scans})

    recent_scans = DetectionHistory.objects.all().order_by('-timestamp')[:10]

    return render(request, 'admin_monitoring.html', {
        'model_info': model_info,
        'total_scans': total_scans,
        'total_users_with_scans': total_users_with_scans,
        'healthy_count': healthy_count,
        'abnormal_count': abnormal_count,
        'disease_distribution': disease_distribution,
        'confidence_stats': confidence_stats,
        'scans_last_7_days': scans_last_7_days,
        'scans_last_30_days': scans_last_30_days,
        'top_patients': top_patients,
        'recent_scans': recent_scans,
    })

# ============== AUDIT LOG ==============

def admin_audit_log(request):
    if 'user_id' not in request.session or request.session.get('role') != 'admin':
        return redirect('home')

    q = request.GET.get('q', '').strip()
    action_filter = request.GET.get('action', '').strip()
    role_filter = request.GET.get('role', '').strip()
    date_filter = request.GET.get('date', '').strip()

    logs_qs = AuditLog.objects.all().select_related('user')

    if q:
        logs_qs = logs_qs.filter(
            Q(user_email__icontains=q) |
            Q(description__icontains=q) |
            Q(ip_address__icontains=q)
        )
    if action_filter:
        logs_qs = logs_qs.filter(action=action_filter)
    if role_filter in ('patient', 'doctor', 'admin'):
        logs_qs = logs_qs.filter(user_role=role_filter)

    now = timezone.now()
    if date_filter == 'today':
        logs_qs = logs_qs.filter(timestamp__gte=now - timedelta(days=1))
    elif date_filter == '7days':
        logs_qs = logs_qs.filter(timestamp__gte=now - timedelta(days=7))
    elif date_filter == '30days':
        logs_qs = logs_qs.filter(timestamp__gte=now - timedelta(days=30))

    total_count = logs_qs.count()
    logs = logs_qs[:500]

    action_summary = {}
    for code, label in AuditLog.ACTION_CHOICES:
        c = AuditLog.objects.filter(action=code).count()
        if c > 0:
            action_summary[code] = c

    return render(request, 'admin_audit_log.html', {
        'logs': logs,
        'q': q,
        'action_filter': action_filter,
        'role_filter': role_filter,
        'date_filter': date_filter,
        'total_count': total_count,
        'action_choices': AuditLog.ACTION_CHOICES,
        'action_summary': action_summary,
    })

def admin_audit_log_clear(request):
    if 'user_id' not in request.session or request.session.get('role') != 'admin':
        return redirect('home')
    if request.method != 'POST':
        return redirect('admin_audit_log')

    days = request.POST.get('days', '90')
    try:
        days = int(days)
    except ValueError:
        days = 90

    if days < 1:
        messages.error(request, "Please enter a valid number of days.")
        return redirect('admin_audit_log')

    cutoff = timezone.now() - timedelta(days=days)
    old_logs = AuditLog.objects.filter(timestamp__lt=cutoff)
    count = old_logs.count()
    old_logs.delete()
    log_action(request, 'other', f"Purged {count} audit logs older than {days} days")
    messages.success(request, f"{count} audit log entries older than {days} days deleted.")
    return redirect('admin_audit_log')