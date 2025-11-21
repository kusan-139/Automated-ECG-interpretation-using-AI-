"""
api/app.py

Unified Flask backend + frontend server
Loads ML models AND serves HTML/CSS/JS frontend files.

IMPORTANT SAFETY NOTICE:
- This tool is for EDUCATIONAL AND RESEARCH purposes only.
- NOT for clinical or diagnostic use.
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.inference.predict import (
    PTBXLInferenceModel,
    MITBIHInferenceModel,
)

app = Flask(
    __name__,
    static_folder="../frontend/static",
    template_folder="../frontend",
)
CORS(app)

# =============================================================
# Load models (Safe fallback)
# =============================================================

ptbxl_model_path = "models/ptbxl_best.pth"
mitbih_model_path = "models/mitbih_best.pth"

ptbxl_model = None
mitbih_model = None

try:
    if os.path.exists(ptbxl_model_path):
        ptbxl_model = PTBXLInferenceModel(ptbxl_model_path)
        print("[INFO] PTB-XL model loaded.")
    else:
        print("[INFO] PTB-XL model NOT found.")
except Exception as e:
    print(f"[ERROR] PTB-XL model load failed: {e}")

try:
    if os.path.exists(mitbih_model_path):
        mitbih_model = MITBIHInferenceModel(mitbih_model_path)
        print("[INFO] MIT-BIH model loaded.")
    else:
        print("[INFO] MIT-BIH model NOT found.")
except Exception as e:
    print(f"[ERROR] MIT-BIH model load failed: {e}")

# =============================================================
# FRONTEND PAGES
# =============================================================


@app.route("/")
def serve_index():
    return send_from_directory("../frontend", "index.html")


""" @app.route("/ptbxl.html")
def serve_ptbxl():
    return send_from_directory("../frontend", "ptbxl.html") """


@app.route("/mitbih.html")
def serve_mitbih():
    return send_from_directory("../frontend", "mitbih.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("../frontend/static", filename)


@app.route("/ptbxl.html")
def serve_ptbxl_stacked():
    return send_from_directory("../frontend", "ptbxl_stacked.html")



# =============================================================
# HEALTH CHECK
# =============================================================


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "ptbxl_model_loaded": ptbxl_model is not None,
            "mitbih_model_loaded": mitbih_model is not None,
            "message": "Research prototype only. NOT for clinical or diagnostic use.",
        }
    )


# =============================================================
# PTB-XL Prediction API  (CSV + JSON)
# =============================================================


@app.route("/api/predict_ecg_ptbxl", methods=["POST"])
def api_predict_ptbxl():
    if ptbxl_model is None:
        return jsonify({"error": "PTB-XL model not loaded"}), 500

    # ---- CASE 1: CSV file upload ----
    if "file" in request.files:
        try:
            f = request.files["file"]
            content = f.read().decode("utf-8")

            ecg_np = np.loadtxt(io.StringIO(content), delimiter=",")
            # ❌ no slicing – keep all 12 columns
            result = ptbxl_model.predict(ecg_np)


            return jsonify(
                {
                    "model": "PTB-XL",
                    "prediction_id": result.get("prediction_id"),
                    "prediction_name": result.get("prediction_name"),
                    # keep "prediction" too in case other code uses it
                    "prediction": result.get("prediction_name"),
                    "probabilities": result.get("probabilities", []),
                    "message": "Research prototype only. NOT for clinical or diagnostic use.",
                }
            )

        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 400

    # ---- CASE 2: JSON body with 'ecg' array ----
    try:
        data = request.get_json()
        if not data or "ecg" not in data:
            raise ValueError("Missing 'ecg' field in JSON body")

        ecg_list = data["ecg"]
        ecg_np = np.array(ecg_list, dtype=float)

        result = ptbxl_model.predict(ecg_np)

        return jsonify(
            {
                "model": "PTB-XL",
                "prediction_id": result.get("prediction_id"),
                "prediction_name": result.get("prediction_name"),
                "prediction": result.get("prediction_name"),
                "probabilities": result.get("probabilities", []),
                "message": "Research prototype only. NOT for clinical or diagnostic use.",
            }
        )

    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


# =============================================================
# MIT-BIH Prediction API  (CSV + JSON)
# =============================================================

@app.route("/api/predict_ecg_mitbih", methods=["POST"])
def api_predict_mitbih():
    if mitbih_model is None:
        return jsonify({"error": "MIT-BIH model not loaded"}), 500

    # ---- CASE 1: CSV file upload ----
    if "file" in request.files:
        try:
            f = request.files["file"]
            content = f.read().decode("utf-8")

            ecg_np = np.loadtxt(io.StringIO(content), delimiter=",")  # (T, 2) expected
            # ❌ no slicing – keep all columns

            result = mitbih_model.predict(ecg_np)

            return jsonify(
                {
                    "model": "MIT-BIH",
                    "prediction_id": result.get("prediction_id"),
                    "prediction_name": result.get("prediction_name"),
                    "prediction": result.get("prediction_name"),
                    "probabilities": result.get("probabilities", []),
                    "message": "Research prototype only. NOT for clinical or diagnostic use.",
                }
            )

        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 400

    # ---- CASE 2: JSON body ----
    try:
        data = request.get_json()
        if not data or "ecg" not in data:
            raise ValueError("Missing 'ecg' field in JSON body")

        ecg_list = data["ecg"]
        ecg_np = np.array(ecg_list, dtype=float)

        result = mitbih_model.predict(ecg_np)

        return jsonify(
            {
                "model": "MIT-BIH",
                "prediction_id": result.get("prediction_id"),
                "prediction_name": result.get("prediction_name"),
                "prediction": result.get("prediction_name"),
                "probabilities": result.get("probabilities", []),
                "message": "Research prototype only. NOT for clinical or diagnostic use.",
            }
        )

    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


# =============================================================
# RUN SERVER
# =============================================================


if __name__ == "__main__":
    print("-------------------------------------------------------")
    print("  Flask backend + frontend running on http://127.0.0.1:5000")
    print("  This tool is NOT for clinical or diagnostic use.")
    print("-------------------------------------------------------")
    app.run(debug=True, host="127.0.0.1", port=5000)
