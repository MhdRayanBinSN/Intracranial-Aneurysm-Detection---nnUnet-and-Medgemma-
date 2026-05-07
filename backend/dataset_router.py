"""
Dataset Information Router — Modular FastAPI router.
Provides RSNA Intracranial Aneurysm Detection dataset statistics.

Added to main.py with:
    from dataset_router import router as dataset_router
    app.include_router(dataset_router)
"""

from fastapi import APIRouter
from pathlib import Path
import os
import json

router = APIRouter(prefix="/dataset", tags=["Dataset"])

DATA_ROOT  = Path("C:/Users/Rayan/Desktop/Main Project")
TRAIN_CSV  = DATA_ROOT / "train.csv"
ZIP_FILE   = DATA_ROOT / "rsna-intracranial-aneurysm-detection.zip"
EVAL_JSON  = Path(__file__).parent / "evaluation_results.json"

LOCATION_COLUMNS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]


@router.get("/info")
def get_dataset_info():
    """Return full RSNA dataset statistics from train.csv."""
    try:
        import pandas as pd

        if not TRAIN_CSV.exists():
            return {"available": False, "error": f"train.csv not found at {TRAIN_CSV}"}

        df = pd.read_csv(TRAIN_CSV)

        total  = len(df)
        pos    = int(df["Aneurysm Present"].sum())
        neg    = total - pos

        modality_counts = df["Modality"].value_counts().to_dict()
        sex_counts      = df["PatientSex"].value_counts().to_dict()

        age_buckets = [
            {"label": "<30",   "count": int((df["PatientAge"] < 30).sum())},
            {"label": "30-39", "count": int(((df["PatientAge"] >= 30) & (df["PatientAge"] < 40)).sum())},
            {"label": "40-49", "count": int(((df["PatientAge"] >= 40) & (df["PatientAge"] < 50)).sum())},
            {"label": "50-59", "count": int(((df["PatientAge"] >= 50) & (df["PatientAge"] < 60)).sum())},
            {"label": "60-69", "count": int(((df["PatientAge"] >= 60) & (df["PatientAge"] < 70)).sum())},
            {"label": "70-79", "count": int(((df["PatientAge"] >= 70) & (df["PatientAge"] < 80)).sum())},
            {"label": "80+",   "count": int((df["PatientAge"] >= 80).sum())},
        ]

        location_data = []
        for col in LOCATION_COLUMNS:
            if col in df.columns:
                count = int(df[col].sum())
                location_data.append({
                    "name": col, "count": count,
                    "pct": round(count / pos * 100, 1) if pos > 0 else 0
                })
        location_data.sort(key=lambda x: x["count"], reverse=True)

        zip_size_gb = round(os.path.getsize(ZIP_FILE) / 1e9, 2) if ZIP_FILE.exists() else None

        return {
            "available": True,
            "overview": {
                "total_series": total,
                "positive_cases": pos,
                "negative_cases": neg,
                "positive_pct": round(pos / total * 100, 1),
                "negative_pct": round(neg / total * 100, 1),
                "total_locations": len(LOCATION_COLUMNS),
            },
            "modality": modality_counts,
            "sex": sex_counts,
            "age": {
                "mean": round(float(df["PatientAge"].mean()), 1),
                "median": round(float(df["PatientAge"].median()), 1),
                "min": int(df["PatientAge"].min()),
                "max": int(df["PatientAge"].max()),
                "buckets": age_buckets,
            },
            "locations": location_data,
            "files": {"zip_size_gb": zip_size_gb},
            "competition": {
                "name": "RSNA 2023 Intracranial Aneurysm Detection",
                "host": "Radiological Society of North America (RSNA)",
                "year": 2023,
                "platform": "Kaggle",
                "task": "Multi-label binary classification across 13 anatomical locations",
                "imaging": "CT Angiography (CTA), MR Angiography (MRA), MRI",
            },
        }

    except Exception as e:
        return {"available": False, "error": str(e)}


@router.get("/evaluation")
def get_evaluation_results():
    """
    Serve the evaluation results produced by evaluate_from_zip.py.
    Returns real TP/FP/FN/TN metrics computed from model inference on the dataset.
    Run: python evaluate_from_zip.py  to generate evaluation_results.json
    """
    if not EVAL_JSON.exists():
        return {
            "available": False,
            "message": "No evaluation results yet. Run: python evaluate_from_zip.py",
        }
    try:
        with open(EVAL_JSON, 'r') as f:
            data = json.load(f)
        data["available"] = True
        return data
    except Exception as e:
        return {"available": False, "error": str(e)}
