DentIQ - AI-Assisted Dental Diagnosis and Explainability System

A web-based dental screening platform that detects four dental conditions from colour intraoral photographs using YOLOv8, with explainability heatmaps and a complete clinical workflow.

Final Year Design Project (FYDP)
Department of Computer Science and IT, University of Southern Punjab, Multan.

Team: Anum Bibi (BSIT-023R22P-05), Sidra (BSIT-023R22P-22), Narjis Fatima (BSIT-023R22P-25)
Supervisor: Sir Zohair Haider

OVERVIEW
========================================
Dental caries affects approximately 2.5 billion people globally, and in Pakistan, prevalence reaches 97.7% among school adolescents. Yet no widely accessible screening tool exists for patients or small clinics.

DentIQ addresses this gap:
1. Multi-class detection: Identifies four conditions (caries, calculus, hypodontia, tooth discoloration) from colour intraoral photographs.
2. Explainability: Provides bounding boxes, confidence tiers, and heatmaps showing where the model is looking.
3. Clinical workflow: Includes patient verification, appointment booking, patient files, medical certificates, audit logging, and model monitoring.

The system runs on CPU only (2 to 5 seconds per image), making it deployable in low-resource settings.

KEY FEATURES
========================================
AI Pipeline:
- YOLOv8s object detection trained on 22,063 colour intraoral images across 5 classes.
- SAHI (Slicing Aided Hyper Inference) for small-lesion detection.
- YOLOv8-seg for pixel-level caries segmentation (runs only when caries is detected).
- Heatmap explainability highlighting regions the model considered important.
- Image quality gating: Laplacian variance blur check and CLIP-based teeth relevance check.

Clinical Workflow:
- Three user roles: Patient, Doctor, Administrator.
- Patient verification gate (two-signal filter) prevents fake appointments.
- Appointment booking with role-aware status transitions.
- Patient File module with unique patient ID (PT-YYYY-NNNN) linking scans, visits, prescriptions, documents, and certificates.
- Medical certificate issuance with downloadable PDF.

Operational Infrastructure:
- 27-action audit log: Immutable record of every security and clinical action with IP address and timestamp.
- Model monitoring dashboard: Per-class detection distribution, confidence statistics, activity timeline.
- Role-based dashboards: Separate interfaces for patients, doctors, and administrators.

MODEL PERFORMANCE
========================================
Evaluated on a 906-image validation set:

Class: Caries
Precision: 0.75, Recall: 0.72, mAP@0.5: 0.79, mAP@0.5:0.95: 0.41

Class: Calculus
Precision: 0.78, Recall: 0.76, mAP@0.5: 0.81, mAP@0.5:0.95: 0.42

Class: Hypodontia
Precision: 0.81, Recall: 0.79, mAP@0.5: 0.84, mAP@0.5:0.95: 0.45

Class: Tooth Discoloration
Precision: 0.78, Recall: 0.79, mAP@0.5: 0.82, mAP@0.5:0.95: 0.43

All classes combined
Precision: 0.780, Recall: 0.766, mAP@0.5: 0.816, mAP@0.5: 0.95: 0.428

Training configuration: YOLOv8s, 30 epochs (early stopping), batch 16, 640x640, AdamW, Google Colab T4 GPU, train-validation loss gap +0.15 (no significant overfitting).

TECH STACK
========================================
Backend: Django 5.2
Language: Python 3.13
Deep Learning: PyTorch 2.0+
Detection: YOLOv8s
Segmentation: YOLOv8-seg
Small Objects: SAHI
Relevance Check: OpenAI CLIP (ViT-B/32)
Explainability: Grad-CAM
Image Processing: OpenCV 4.5+
PDF Generation: ReportLab 3.6+
Database: SQLite
Frontend: Bootstrap 5.3

INSTALLATION
========================================
Prerequisites:
- Python 3.13 or later
- pip
- 4 GB RAM minimum (8 GB recommended)

Steps:

1. Clone the repository
git clone https://github.com/sidrahkhan2204-bot/DentiQ-AI.git
cd DentiQ-AI

2. Create a virtual environment and install dependencies
python -m venv venv
On Windows: venv\Scripts\activate
On Linux or Mac: source venv/bin/activate
pip install -r requirements.txt

3. Download model weights
The model weights (.pt files) are too large for GitHub. Download them from the links below and place them in a new folder named "models" in the root of your project.
- Detection model (detect_augmented_final.pt): https://drive.google.com/file/d/1hH7mpA2EPRo8WHxNvIXEh5RzSyyoLtDX/view?usp=sharing
- Segmentation model (seg_augmented_final.pt): https://drive.google.com/file/d/1SlJ1ZBCfI9vU0HbpyTgZ3_OwJoZHMOF1/view?usp=sharing

4. Create media directories
mkdir -p media/uploads media/patient_docs media/reports media/runs/seg

5. Apply database migrations and run
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver

Navigate to http://127.0.0.1:8000/ in your browser.

PROJECT STRUCTURE
========================================
DentiQ-AI/
ai_app/ (Core application: views, models, urls)
dental_project/ (Django project settings)
templates/ (HTML templates)
database/ (SQLite database)
media/ (Uploaded images and generated PDFs)
requirements.txt
manage.py
README.md
