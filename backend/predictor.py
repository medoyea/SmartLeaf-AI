"""
SmartLeaf AI — Predictor Module
================================
Handles model loading and inference for both:
  - CNN (EfficientNet-B0 fine-tuned, end-to-end)
  - SVM (EfficientNet-B0 features + SVM classifier)

Also includes OpenCV-based severity estimation.
"""

import os
import json
import time
import logging
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger("SmartLeafAI.Predictor")

# ─── Demo Class Labels (used when real models are absent) ─────────────────────
DEMO_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _build_cnn_model(num_classes: int):
    """
    Rebuild EfficientNet-B0 classifier architecture.
    Must match exactly what was used during training so weights load correctly.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model

    base = EfficientNetB0(
        weights=None,           # weights loaded separately
        include_top=False,
        input_shape=(224, 224, 3),
    )
    inputs  = base.input
    x       = base.output
    x       = GlobalAveragePooling2D(name="gap")(x)
    x       = BatchNormalization()(x)
    x       = Dense(256, activation="relu", name="dense_head")(x)
    x       = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model   = Model(inputs, outputs, name="SmartLeaf_EfficientNetB0")
    return model, base


def _build_feature_extractor(num_classes: int):
    """
    Build feature extractor (EfficientNet → GAP → 1280-dim vector).
    Shares the same base as the CNN classifier.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    inputs    = base.input
    x         = GlobalAveragePooling2D(name="feature_pool")(base.output)
    extractor = Model(inputs, x, name="EfficientNetB0_Extractor")
    return extractor


# ─── SmartLeafPredictor ───────────────────────────────────────────────────────

class SmartLeafPredictor:
    """
    Unified predictor that runs both CNN and SVM inference pipelines
    and performs OpenCV-based disease severity estimation.
    """

    IMG_SIZE = (224, 224)

    def __init__(
        self,
        cnn_model_path: Optional[str] = None,
        svm_model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        demo_mode: bool = False,
    ):
        self.demo_mode   = demo_mode
        self.models_ready = False
        self.cnn_model   = None
        self.svm_model   = None
        self.scaler      = None
        self.feature_extractor = None
        self.class_labels: list = DEMO_LABELS.copy()

        if not demo_mode:
            self._load_labels(labels_path)
            self._load_models(cnn_model_path, svm_model_path)
        else:
            logger.warning("Running in DEMO MODE — using simulated predictions.")

    # ── Initialization ────────────────────────────────────────────────────────

    def _load_labels(self, labels_path: Optional[str]):
        """Load class labels from JSON file."""
        if labels_path and Path(labels_path).exists():
            with open(labels_path, "r") as f:
                data = json.load(f)
            self.class_labels = data if isinstance(data, list) else data.get("labels", DEMO_LABELS)
            logger.info(f"Loaded {len(self.class_labels)} class labels")
        else:
            logger.warning("Labels file not found — using demo labels")
            self.class_labels = DEMO_LABELS.copy()

    def _load_models(self, cnn_path: Optional[str], svm_path: Optional[str]):
        """
        Load trained CNN and SVM models.
        Supports both:
          - weights-only .h5 files (saved with model.save_weights())
          - full SavedModel .h5 files (saved with model.save())
        """
        try:
            import tensorflow as tf

            num_classes = len(self.class_labels)

            # ── Load CNN ──────────────────────────────────────────────────────
            if cnn_path and Path(cnn_path).exists():
                # To this:
                self.cnn_model, _ = _build_cnn_model(num_classes)
                self.cnn_model.load_weights(cnn_path)
                logger.info(f"CNN weights loaded: {cnn_path}")
            else:
                logger.warning(f"CNN model not found at {cnn_path}. Train first.")

            # ── Build feature extractor from loaded CNN ───────────────────────
            if self.cnn_model is not None:
                # Feature extractor shares input with CNN, outputs 256-dim head vector
                # We use the layer just before softmax for richer SVM features
                inputs = self.cnn_model.input
                feat_layer = self.cnn_model.get_layer("gap").output
                self.feature_extractor = tf.keras.Model(inputs, feat_layer)
                logger.info("Feature extractor built from CNN (dense_head layer)")

            # ── Load SVM + scaler ─────────────────────────────────────────────
            if svm_path and Path(svm_path).exists():
                with open(svm_path, "rb") as f:
                    self.svm_model = pickle.load(f)
                logger.info(f"SVM model loaded: {svm_path}")

                # Load scaler if present alongside svm
                scaler_path = Path(svm_path).parent / "feature_scaler.pkl"
                if scaler_path.exists():
                    with open(str(scaler_path), "rb") as f:
                        self.scaler = pickle.load(f)
                    logger.info("Feature scaler loaded")
            else:
                logger.warning(f"SVM model not found at {svm_path}. Train first.")

            if self.cnn_model and self.svm_model:
                self.models_ready = True
                self.demo_mode    = False
                logger.info("All models ready for inference")
            else:
                logger.warning("One or more models missing — falling back to demo mode")
                self.demo_mode = True

        except ImportError as e:
            logger.error(f"TensorFlow not available: {e}. Using demo mode.")
            self.demo_mode = True
        except Exception as e:
            logger.error(f"Error loading models: {e}. Using demo mode.")
            self.demo_mode = True

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for EfficientNet inference.
        Uses efficientnet.preprocess_input (scales to -1..1 range),
        matching the preprocessing used during training.
        Returns float32 array of shape (1, 224, 224, 3).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.IMG_SIZE, interpolation=cv2.INTER_AREA)

        # Apply EfficientNet preprocessing (same as training pipeline)
        try:
            import tensorflow as tf
            img_array = tf.keras.applications.efficientnet.preprocess_input(
                img_resized.astype(np.float32)
            )
        except Exception:
            # Fallback: manual scale to [-1, 1]
            img_array = (img_resized.astype(np.float32) / 127.5) - 1.0

        return np.expand_dims(img_array, axis=0)

    # ── CNN Inference ─────────────────────────────────────────────────────────

    def predict_cnn(self, image_path: str) -> dict:
        """Run end-to-end CNN prediction."""
        if self.demo_mode or self.cnn_model is None:
            return self._demo_prediction(image_path, model="cnn")

        start = time.time()
        img   = self.preprocess_image(image_path)
        probs = self.cnn_model.predict(img, verbose=0)[0]
        elapsed_ms = (time.time() - start) * 1000

        top_k_indices = np.argsort(probs)[::-1][:5]
        top_k = [
            {"class": self.class_labels[i], "confidence": float(probs[i])}
            for i in top_k_indices
        ]

        return {
            "class":        self.class_labels[int(np.argmax(probs))],
            "confidence":   float(np.max(probs)),
            "top_k":        top_k,
            "inference_ms": round(elapsed_ms, 2),
        }

    # ── SVM Inference ─────────────────────────────────────────────────────────

    def predict_svm(self, image_path: str) -> dict:
        """Extract CNN features and classify with SVM."""
        if self.demo_mode or self.feature_extractor is None or self.svm_model is None:
            return self._demo_prediction(image_path, model="svm")

        start    = time.time()
        img      = self.preprocess_image(image_path)
        features = self.feature_extractor.predict(img, verbose=0)
        features_flat = features.reshape(1, -1)

        # Apply scaler if available
        if self.scaler is not None:
            features_flat = self.scaler.transform(features_flat)

        predicted_idx = self.svm_model.predict(features_flat)[0]
        elapsed_ms    = (time.time() - start) * 1000

        confidence = 0.85
        if hasattr(self.svm_model, "predict_proba"):
            try:
                proba      = self.svm_model.predict_proba(features_flat)[0]
                confidence = float(np.max(proba))
            except Exception:
                pass

        # Handle both int index and string class name from SVM
        if isinstance(predicted_idx, (int, np.integer)):
            predicted_class = self.class_labels[int(predicted_idx)]
        else:
            predicted_class = str(predicted_idx)

        return {
            "class":        predicted_class,
            "confidence":   confidence,
            "inference_ms": round(elapsed_ms, 2),
        }

    # ── Severity Estimation (OpenCV) ──────────────────────────────────────────

    def estimate_severity(self, image_path: str) -> dict:
        """
        Use OpenCV HSV color analysis to estimate disease severity.
        Analyzes brown/yellow/dark pixel percentage as proxy for infection area.

        Returns:
            level              : 'healthy' | 'mild' | 'moderate' | 'severe'
            score              : 0–100
            affected_percentage: estimated % of image area showing disease
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"level": "unknown", "score": 0, "affected_percentage": 0}

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Healthy green mask
            leaf_mask = cv2.inRange(img_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

            # Disease colors
            brown_mask  = cv2.inRange(img_hsv, np.array([10, 40, 40]),  np.array([30, 255, 180]))
            yellow_mask = cv2.inRange(img_hsv, np.array([22, 60, 100]), np.array([38, 255, 255]))

            # Dark lesions
            _, dark_mask = cv2.threshold(img_hsv[:, :, 2], 60, 255, cv2.THRESH_BINARY_INV)
            dark_mask    = cv2.bitwise_and(dark_mask, leaf_mask)

            disease_mask = cv2.bitwise_or(brown_mask, yellow_mask)
            disease_mask = cv2.bitwise_or(disease_mask, dark_mask)

            total_pixels   = img.shape[0] * img.shape[1]
            disease_pixels = cv2.countNonZero(disease_mask)
            affected_pct   = (disease_pixels / total_pixels) * 100
            score          = min(int(affected_pct * 2.5), 100)

            if affected_pct < 5:
                level = "healthy"
            elif affected_pct < 15:
                level = "mild"
            elif affected_pct < 35:
                level = "moderate"
            else:
                level = "severe"

            r_mean = float(np.mean(img_rgb[:, :, 0]))
            g_mean = float(np.mean(img_rgb[:, :, 1]))
            b_mean = float(np.mean(img_rgb[:, :, 2]))

            return {
                "level":              level,
                "score":              score,
                "affected_percentage": round(affected_pct, 1),
                "color_analysis": {
                    "r_mean":           round(r_mean, 1),
                    "g_mean":           round(g_mean, 1),
                    "b_mean":           round(b_mean, 1),
                    "greenness_index":  round(g_mean - max(r_mean, b_mean), 1),
                },
                "pixel_stats": {
                    "total":           total_pixels,
                    "leaf_area":       cv2.countNonZero(leaf_mask),
                    "disease_pixels":  disease_pixels,
                },
            }

        except Exception as e:
            logger.error(f"Severity estimation error: {e}")
            return {"level": "unknown", "score": 0, "affected_percentage": 0}

    # ── Combined Prediction Pipeline ──────────────────────────────────────────

    def predict_full(self, image_path: str) -> dict:
        """
        Run complete prediction pipeline:
          1. CNN end-to-end prediction
          2. CNN features + SVM prediction
          3. OpenCV severity estimation
        """
        cnn_result = self.predict_cnn(image_path)
        svm_result = self.predict_svm(image_path)
        severity   = self.estimate_severity(image_path)

        return {
            "cnn":      cnn_result,
            "svm":      svm_result,
            "severity": severity,
        }

    # ── Demo Mode ─────────────────────────────────────────────────────────────

    def _demo_prediction(self, image_path: str, model: str = "cnn") -> dict:
        """
        Return realistic demo prediction when models are not loaded.
        Uses image path hash for deterministic outputs.
        """
        seed = abs(hash(image_path)) % 1000
        rng  = random.Random(seed)

        label      = rng.choice(self.class_labels)
        confidence = rng.uniform(0.75, 0.99)

        if model == "cnn":
            top_k     = [{"class": label, "confidence": round(confidence, 4)}]
            remaining = 1.0 - confidence
            others    = [l for l in self.class_labels if l != label]
            rng.shuffle(others)
            for other in others[:4]:
                share      = rng.uniform(0, remaining * 0.5)
                remaining -= share
                top_k.append({"class": other, "confidence": round(share, 4)})
            return {
                "class":        label,
                "confidence":   round(confidence, 4),
                "top_k":        top_k,
                "inference_ms": round(rng.uniform(20, 50), 1),
                "demo":         True,
            }

        svm_label = label if rng.random() > 0.15 else rng.choice(self.class_labels)
        return {
            "class":        svm_label,
            "confidence":   round(rng.uniform(0.70, 0.95), 4),
            "inference_ms": round(rng.uniform(35, 70), 1),
            "demo":         True,
        }