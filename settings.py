"""
Django settings for dental_project.

DentIQ — AI-Powered Dental Diagnostics
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# SECURITY
# ============================================================
SECRET_KEY = 'django-insecure-dentiq-fyp-dental-project-2026-change-me'

DEBUG = True

# Updated to allow your local computer, mobile phone, and ngrok
ALLOWED_HOSTS = ['192.168.1.12', 'localhost', '127.0.0.1', 'unspoken-distill-extrude.ngrok-free.dev', '*']


# ============================================================
# CSRF TRUSTED ORIGINS (Needed for Mobile & Ngrok Access)
# ============================================================
CSRF_TRUSTED_ORIGINS = [
    'http://192.168.1.12:8000',
    'https://unspoken-distill-extrude.ngrok-free.dev',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
]


# ============================================================
# APPLICATIONS
# ============================================================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ai_app',
]


# ============================================================
# MIDDLEWARE
# ============================================================
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# ============================================================
# URLS & WSGI
# ============================================================
ROOT_URLCONF = 'dental_project.urls'

WSGI_APPLICATION = 'dental_project.wsgi.application'


# ============================================================
# TEMPLATES
# ============================================================
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # DentIQ custom: adds patient_is_verified to every template
                'ai_app.context_processors.verification_status',
            ],
        },
    },
]


# ============================================================
# DATABASE
# ============================================================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'database' / 'dental.db',
    }
}


# ============================================================
# PASSWORD VALIDATION
# ============================================================
AUTH_PASSWORD_VALIDATORS = [
    # Disabled for FYP demo — patients use simple passwords.
    # Re-enable in production.
]


# ============================================================
# INTERNATIONALIZATION
# ============================================================
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Karachi'
USE_I18N = True
USE_TZ = True


# ============================================================
# STATIC & MEDIA FILES
# ============================================================
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'


# ============================================================
# DEFAULT PRIMARY KEY
# ============================================================
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ============================================================
# SESSION
# ============================================================
SESSION_COOKIE_AGE = 60 * 60 * 24  # 24 hours
SESSION_EXPIRE_AT_BROWSER_CLOSE = False