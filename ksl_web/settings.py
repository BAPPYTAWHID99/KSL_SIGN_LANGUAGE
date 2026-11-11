from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = "local-development-only-change-before-deployment"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "ksl_web",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "ksl_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]

WSGI_APPLICATION = "ksl_web.wsgi.application"
ASGI_APPLICATION = "ksl_web.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
DATA_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024
