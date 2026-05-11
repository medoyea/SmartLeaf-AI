"""
SmartLeaf AI - Flask Backend Application
=========================================
Main Flask application providing REST API endpoints for plant disease
detection, treatment recommendations, and model comparison.
"""

import os
import sys
import json
import time
import base64
import logging
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import SmartLeafPredictor

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR.parent / "results"
TREATMENT_DB_PATH = BASE_DIR / "treatment_data.json"
MODEL_CNN_PATH = BASE_DIR.parent / "models" / "efficientnet_finetuned.h5"
MODEL_SVM_PATH = BASE_DIR.parent / "models" / "svm_classifier.pkl"
LABELS_PATH = BASE_DIR.parent / "models" / "class_labels.json"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

# ─── App Initialization ─────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "smartleaf.log"),
    ],
)
logger = logging.getLogger("SmartLeafAI")

# ─── Load Treatment Database ────────────────────────────────────────────────────

with open(TREATMENT_DB_PATH, "r", encoding="utf-8") as f:
    TREATMENT_DB = json.load(f)
logger.info(f"Loaded treatment database: {len(TREATMENT_DB)} disease entries")

# ─── Initialize Predictor ───────────────────────────────────────────────────────

predictor = None

def get_predictor():
    """Lazy-load the predictor singleton."""
    global predictor
    if predictor is None:
        try:
            predictor = SmartLeafPredictor(
                cnn_model_path=str(MODEL_CNN_PATH),
                svm_model_path=str(MODEL_SVM_PATH),
                labels_path=str(LABELS_PATH),
            )
            logger.info("SmartLeaf predictor initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load trained models: {e}. Running in demo mode.")
            predictor = SmartLeafPredictor(demo_mode=True)
    return predictor

# ─── Helper Functions ───────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_treatment_info(class_name: str) -> dict:
    """Retrieve treatment information for a given class name."""
    if class_name in TREATMENT_DB:
        return TREATMENT_DB[class_name]
    # Try fuzzy match
    for key in TREATMENT_DB:
        if key.lower().replace("_", " ") in class_name.lower():
            return TREATMENT_DB[key]
    # Return default healthy advice if not found
    return {
        "common_name": class_name.replace("_", " ").replace("___", " — "),
        "plant": class_name.split("___")[0] if "___" in class_name else "Unknown",
        "description": "Disease information not available in our database.",
        "causes": ["Consult local agricultural extension service"],
        "symptoms": ["See a plant pathologist for accurate diagnosis"],
        "treatment": ["Consult a local agricultural expert"],
        "prevention": ["Practice good agricultural hygiene"],
        "watering_advice": "Follow standard practices for your plant type.",
        "fertilizer_recommendation": "Apply balanced fertilizer according to soil test results.",
        "severity_tips": {
            "mild": "Monitor and consult an expert.",
            "moderate": "Seek professional advice promptly.",
            "severe": "Contact agricultural extension service immediately.",
        },
    }


def is_healthy_class(class_name: str) -> bool:
    return "healthy" in class_name.lower()

# ─── Scan History (In-Memory Store) ─────────────────────────────────────────────

scan_history = []
stats = {
    "total_scans": 0,
    "healthy_count": 0,
    "diseased_count": 0,
    "disease_distribution": {},
    "avg_confidence": 0.0,
}

# ─── API Endpoints ───────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    """Serve the frontend index page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    p = get_predictor()
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "models_loaded": p.models_ready,
        "demo_mode": p.demo_mode,
        "treatment_db_entries": len(TREATMENT_DB),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Accepts: multipart/form-data with 'image' file OR JSON with 'image_base64'
    Returns: prediction results from both CNN and SVM models with treatment info.
    """
    start_time = time.time()

    # ── Accept image ──────────────────────────────────────────────────────────
    image_path = None
    temp_filename = None

    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Supported: {ALLOWED_EXTENSIONS}"}), 400

        ext = file.filename.rsplit(".", 1)[1].lower()
        temp_filename = f"{uuid.uuid4().hex}.{ext}"
        image_path = UPLOAD_FOLDER / temp_filename
        file.save(str(image_path))

    elif request.is_json and "image_base64" in request.json:
        try:
            img_data = base64.b64decode(request.json["image_base64"])
            temp_filename = f"{uuid.uuid4().hex}.jpg"
            image_path = UPLOAD_FOLDER / temp_filename
            with open(str(image_path), "wb") as f:
                f.write(img_data)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
    else:
        return jsonify({"error": "No image provided. Send 'image' file or 'image_base64' field."}), 400

    # ── Run predictions ───────────────────────────────────────────────────────
    try:
        p = get_predictor()
        result = p.predict_full(str(image_path))

        class_name = result["cnn"]["class"]
        confidence = result["cnn"]["confidence"]
        treatment = get_treatment_info(class_name)
        healthy = is_healthy_class(class_name)

        # ── Update stats ──────────────────────────────────────────────────────
        stats["total_scans"] += 1
        if healthy:
            stats["healthy_count"] += 1
        else:
            stats["diseased_count"] += 1
            disease = treatment["common_name"]
            stats["disease_distribution"][disease] = stats["disease_distribution"].get(disease, 0) + 1

        n = stats["total_scans"]
        stats["avg_confidence"] = ((stats["avg_confidence"] * (n - 1)) + confidence) / n

        # ── Build history entry ───────────────────────────────────────────────
        scan_entry = {
            "id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().isoformat(),
            "disease": treatment["common_name"],
            "plant": treatment["plant"],
            "healthy": healthy,
            "confidence": round(confidence * 100, 1),
            "severity": result.get("severity", "N/A"),
        }
        scan_history.insert(0, scan_entry)
        if len(scan_history) > 50:
            scan_history.pop()

        elapsed = round((time.time() - start_time) * 1000)

        # ── Compose response ──────────────────────────────────────────────────
        response = {
            "success": True,
            "inference_time_ms": elapsed,
            "prediction": {
                "cnn": {
                    "class": class_name,
                    "confidence": round(confidence * 100, 2),
                    "top_predictions": result["cnn"].get("top_k", []),
                },
                "svm": {
                    "class": result["svm"]["class"],
                    "confidence": round(result["svm"]["confidence"] * 100, 2),
                    "agreement": result["svm"]["class"] == result["cnn"]["class"],
                },
                "consensus": class_name,
                "is_healthy": healthy,
            },
            "severity": result.get("severity", {
                "level": "N/A",
                "score": 0,
                "affected_percentage": 0,
                "color_analysis": {},
            }),
            "treatment": treatment,
            "scan_id": scan_entry["id"],
        }

        logger.info(f"Prediction complete: {class_name} ({confidence:.2%}) in {elapsed}ms")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Clean up temp file
        if image_path and image_path.exists():
            try:
                os.remove(str(image_path))
            except Exception:
                pass


@app.route("/api/compare", methods=["GET"])
def compare_models():
    """
    GET /api/compare
    Returns comparison metrics between CNN and SVM approaches
    loaded from saved evaluation results.
    """
    results_file = RESULTS_FOLDER / "comparison_results.json"

    if results_file.exists():
        with open(str(results_file), "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})

    # Return demo comparison data if results not yet generated
    demo_data = {
        "note": "Demo data — run training notebook to generate real metrics",
        "cnn": {
            "name": "EfficientNet-B0 (End-to-End Fine-tuned)",
            "accuracy": 94.2,
            "precision": 93.8,
            "recall": 94.1,
            "f1_score": 93.9,
            "training_time_seconds": 1842,
            "avg_inference_ms": 28,
            "parameters": "5.3M (trainable: 1.2M after partial freeze)",
        },
        "svm": {
            "name": "EfficientNet-B0 Features + SVM (RBF Kernel)",
            "accuracy": 91.7,
            "precision": 91.2,
            "recall": 91.5,
            "f1_score": 91.3,
            "training_time_seconds": 320,
            "avg_inference_ms": 45,
            "parameters": "Feature dim: 1280 + SVM (C=10, gamma=scale)",
        },
        "winner": "CNN",
        "summary": (
            "End-to-End CNN achieves higher accuracy (94.2% vs 91.7%) at faster inference "
            "(28ms vs 45ms). SVM trains significantly faster (320s vs 1842s) and may be "
            "preferred in resource-constrained environments."
        ),
    }
    return jsonify({"success": True, "data": demo_data})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """GET /api/stats — Returns dashboard statistics and scan history."""
    return jsonify({
        "success": True,
        "stats": {
            "total_scans": stats["total_scans"],
            "healthy_count": stats["healthy_count"],
            "diseased_count": stats["diseased_count"],
            "disease_distribution": stats["disease_distribution"],
            "avg_confidence": round(stats["avg_confidence"] * 100, 1),
            "health_rate": round(
                (stats["healthy_count"] / max(stats["total_scans"], 1)) * 100, 1
            ),
        },
        "recent_scans": scan_history[:10],
    })


@app.route("/api/treatments", methods=["GET"])
def list_treatments():
    """GET /api/treatments — Returns all available treatments."""
    treatments = []
    for key, val in TREATMENT_DB.items():
        treatments.append({
            "key": key,
            "common_name": val["common_name"],
            "plant": val["plant"],
            "is_healthy": is_healthy_class(key),
        })
    return jsonify({"success": True, "count": len(treatments), "treatments": treatments})


@app.route("/api/treatments/<disease_key>", methods=["GET"])
def get_treatment(disease_key):
    """GET /api/treatments/<key> — Returns treatment for specific disease."""
    if disease_key in TREATMENT_DB:
        return jsonify({"success": True, "data": TREATMENT_DB[disease_key]})
    return jsonify({"error": "Treatment not found"}), 404


# ─── Error Handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  SmartLeaf AI — Plant Disease Detection System")
    logger.info("  Version 1.0.0  |  Running on http://localhost:5000")
    logger.info("=" * 60)
    get_predictor()  # Pre-warm the predictor
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
