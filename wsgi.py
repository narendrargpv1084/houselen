"""
wsgi.py — Gunicorn entry point
Render start command:  gunicorn wsgi:app
"""
from app import app  # noqa: F401
