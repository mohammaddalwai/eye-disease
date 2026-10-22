FROM python:3.11-slim

# PORT: 7860 is what Hugging Face Spaces expects. Override it with -e PORT=... on other hosts.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

# Run as a normal user, not root. UID 1000 is also what Hugging Face Spaces uses.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Install libraries first. This layer is cached, so code changes rebuild in seconds.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then copy the app, the model and the pages.
COPY --chown=appuser:appuser app.py model.h5 ./
COPY --chown=appuser:appuser templates ./templates
COPY --chown=appuser:appuser static ./static

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=3)"]

# One worker with two threads: TensorFlow is memory-heavy, so we don't run many copies of the model.
# --timeout 120 gives the model time to load at startup.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 --access-logfile - app:app"]
