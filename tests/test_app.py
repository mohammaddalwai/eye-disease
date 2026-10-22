"""Basic checks for the Flask app.

These use made-up images. They test that the app handles uploads correctly,
not how accurate the model is.
"""
import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import CLASSES, MAX_UPLOAD_BYTES, app  # noqa: E402  (importing this loads the model once)

STATIC_DIR = os.path.join(app.root_path, "static")


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def image_bytes(fmt="JPEG", mode="RGB", size=(320, 240)):
    img = Image.effect_noise(size, 60).convert(mode)   # random-looking noise image
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def upload(client, data, filename="eye.jpg"):
    return client.post(
        "/upload_image",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok", "model_loaded": True}


@pytest.mark.parametrize("path", ["/", "/faq", "/symptoms"])
def test_pages_load_and_show_disclaimer(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert b"Not medical advice" in r.data


@pytest.mark.parametrize("fmt,mode,name", [
    ("JPEG", "RGB", "eye.jpg"),
    ("PNG", "RGB", "eye.png"),
    ("PNG", "RGBA", "eye_alpha.png"),   # used to crash the old app
    ("PNG", "L", "eye_gray.png"),       # used to crash the old app
])
def test_valid_upload_returns_a_diagnosis(client, fmt, mode, name):
    r = upload(client, image_bytes(fmt, mode), name)
    assert r.status_code == 200
    assert "data:image/jpeg;base64," in r.get_data(as_text=True)
    assert any(c in r.get_data(as_text=True) for c in CLASSES)
    assert r.headers["Cache-Control"] == "no-store"


def test_upload_is_not_saved_to_disk(client):
    before = sorted(os.listdir(STATIC_DIR))
    assert upload(client, image_bytes()).status_code == 200
    assert sorted(os.listdir(STATIC_DIR)) == before


def test_rejects_text_file_renamed_to_jpg(client):
    assert upload(client, b"this is not an image", "fake.jpg").status_code == 400


def test_rejects_other_image_formats(client):
    assert upload(client, image_bytes("GIF", "L"), "eye.gif").status_code == 415


def test_rejects_oversized_upload(client):
    r = upload(client, b"0" * (MAX_UPLOAD_BYTES + 1024))
    assert r.status_code == 413


def test_rejects_missing_file(client):
    assert client.post("/upload_image", data={}).status_code == 400


def test_rejects_empty_filename(client):
    assert upload(client, image_bytes(), filename="").status_code == 400
