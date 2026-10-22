import base64
import io
import logging
import os
import threading
import time

# Quieter TensorFlow logs. Must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eye_detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model.h5"))

# Order must match the class order used in training (alphabetical folder names).
CLASSES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal", "Other"]
IMG_SIZE = (256, 256)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024        # reject uploads bigger than 5 MB
MAX_IMAGE_PIXELS = 25_000_000             # reject huge images (decompression bombs)
ALLOWED_FORMATS = {"JPEG", "PNG"}         # checked on the real file content, not the file name

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Load the model once, when the server starts (not on every request).
log.info("Loading model from %s", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
# One dummy prediction so the first real user doesn't pay the warm-up cost.
model(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), training=False)
log.info("Model ready")

# One prediction at a time. Other requests wait their turn (about 100 ms each on CPU).
_predict_lock = threading.Lock()

disease_descriptions = {
    'Cataract': "Cataract is a clouding of the eye's lens, which leads to blurred vision. It's often age-related but can also result from injury or other medical conditions. Symptoms include blurry or foggy vision, difficulty seeing at night, sensitivity to light, and seeing halos around lights.",
    'Diabetic Retinopathy': "Diabetic retinopathy is a complication of diabetes that affects the eyes. It occurs when high blood sugar levels damage the blood vessels in the retina, leading to vision problems. Symptoms include blurred vision, floaters, difficulty seeing at night, and vision loss.",
    'Glaucoma': "Glaucoma is a group of eye conditions that damage the optic nerve, often due to increased pressure in the eye. It can lead to vision loss or blindness if left untreated. Symptoms may not appear until the condition is advanced, but they can include blurred vision, eye pain, headache, and seeing halos around lights.",
    'Normal': "In the context of your model, 'Normal' indicates that no signs of eye diseases were detected in the uploaded image. This result suggests that the eyes appear healthy without any abnormalities or conditions requiring immediate attention.",
    'Other': "'Other' typically refers to any eye condition not specifically categorized by the model. It could include less common eye diseases or conditions that the model hasn't been trained to identify. Further evaluation by an eye care professional may be necessary to determine the specific issue."
}

SYMPTOM_FIELDS = [
    'blurredVision', 'halosAroundLights', 'fluctuatingVision',
    'darkSpotsFloaters', 'gradualPeripheralVisionLoss', 'severeEyePain',
]


class BadImage(Exception):
    """The upload is not a usable image. Carries the HTTP status and a message."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def read_image(file_storage):
    """Open the upload, check it is a real JPEG/PNG of sane size, return an RGB image."""
    try:
        img = Image.open(file_storage.stream)   # reads the header only
        fmt = img.format
        width, height = img.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise BadImage("The file is not a valid image.", 400)

    if fmt not in ALLOWED_FORMATS:
        raise BadImage("Only JPEG and PNG images are accepted.", 415)
    if width * height > MAX_IMAGE_PIXELS:
        raise BadImage("The image is too large. Please upload a smaller one.", 400)

    try:
        return img.convert("RGB").resize(IMG_SIZE)   # full decode happens here
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise BadImage("The image file is damaged or cannot be read.", 400)


def predict(img):
    """Return (class name, confidence in percent, milliseconds taken)."""
    batch = np.asarray(img, dtype="float32")[np.newaxis, ...] / 255.0
    start = time.perf_counter()
    with _predict_lock:
        probs = model(batch, training=False).numpy()[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]) * 100, elapsed_ms


def to_data_uri(img):
    """Encode the resized image for display on the result page. Nothing is written to disk."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", model_loaded=True)


@app.route("/", methods=["GET"])
def home():
    return render_template('home.html')


@app.route("/faq", methods=["GET"])
def faq():
    return render_template('faq.html')


@app.route("/symptoms", methods=["GET"])
def symptoms():
    return render_template('symptoms.html')


@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get('file')
    if file is None:
        return "No file part", 400
    if file.filename == '':
        return "No selected file", 400

    try:
        img = read_image(file)
    except BadImage as e:
        return e.message, e.status

    predicted_class, probability, ms = predict(img)
    log.info("prediction class=%s confidence=%.1f%% inference_ms=%.0f",
             predicted_class, probability, ms)

    symptoms = {name: request.form.get(name) for name in SYMPTOM_FIELDS}
    response = app.make_response(render_template(
        'diagnosis.html',
        predicted_class=predicted_class,
        probability=round(probability, 1),
        image_data=to_data_uri(img),
        disease_description=disease_descriptions.get(predicted_class, "Description not available"),
        symptoms=symptoms,
    ))
    response.headers["Cache-Control"] = "no-store"   # don't let browsers or proxies keep the result page
    return response


@app.errorhandler(413)
def too_large(_e):
    return f"The file is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413


if __name__ == "__main__":
    # Local development only. In Docker, gunicorn starts the app.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
