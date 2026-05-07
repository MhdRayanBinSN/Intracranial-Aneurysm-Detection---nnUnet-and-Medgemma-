#!/usr/bin/env python
"""
Batch Evaluation Script for Aneurysm Detection Model

Compares model predictions with ground truth annotations.
Generates accuracy metrics and confusion matrix.

Usage:
    python evaluate_model.py --series-folder <SERIES_FOLDER>
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Import the detector from our CLI script
from detect_aneurysm import AneurysmDetector

# =============================================================================
# GROUND TRUTH DATA (from user's annotation)
# =============================================================================
# Format: (SeriesInstanceUID, Location)
GROUND_TRUTH = [
    ("1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317", "Other Posterior Circulation"),
    ("1.2.826.0.1.3680043.8.498.10022796280698534221758473208024838831", "Right Middle Cerebral Artery"),
    ("1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636", "Right Middle Cerebral Artery"),
    ("1.2.826.0.1.3680043.8.498.10030095840917973694487307992374923817", "Right Infraclinoid Internal Carotid Artery"),
    ("1.2.826.0.1.3680043.8.498.10034081836061566510187499603024895557", "Anterior Communicating Artery"),
    ("1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381", "Right Anterior Cerebral Artery"),
    ("1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381", "Left Middle Cerebral Artery"),
    ("1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381", "Right Supraclinoid Internal Carotid Artery"),
    ("1.2.826.0.1.3680043.8.498.10042423585566957032411171949972906248", "Right Middle Cerebral Artery"),
    ("1.2.826.0.1.3680043.8.498.10042474696169267476037627878420766468", "Left Supraclinoid Internal Carotid Artery"),
    ("1.2.826.0.1.3680043.8.498.10058383541003792190302541266378919328", "Right Middle Cerebral Artery"),
    ("1.2.826.0.1.3680043.8.498.10076056930521523789588901704956188485", "Right Supraclinoid Internal Carotid Artery"),
]

# Build lookup: series_id -> list of locations
GT_LOOKUP = defaultdict(list)
for series_id, location in GROUND_TRUTH:
    GT_LOOKUP[series_id].append(location)

# Series with aneurysms (positive cases)
POSITIVE_SERIES = set(GT_LOOKUP.keys())


def evaluate_single_series(detector, series_path: Path, series_id: str) -> Dict:
    """
    Evaluate detection on a single series.
    
    Returns dict with:
    - ground_truth: list of actual aneurysm locations
    - predictions: dict of location -> probability
    - detections: list of detected locations (>50%)
    - is_positive: whether this series has aneurysm (ground truth)
    - correctly_detected: list of correctly found locations
    - false_positives: list of wrongly detected locations
    - false_negatives: list of missed locations
    """
    gt_locations = GT_LOOKUP.get(series_id, [])
    is_positive = series_id in POSITIVE_SERIES
    
    # Run detection
    result = detector.detect(series_path)
    
    if "error" in result:
        return {
            "series_id": series_id,
            "error": result["error"],
            "ground_truth": gt_locations,
            "is_positive": is_positive,
        }
    
    # Extract predictions
    all_preds = result.get("per_location", {})
    detections = result.get("detections", [])
    detected_locations = [d["location"] for d in detections]
    
    # Calculate metrics
    correctly_detected = [loc for loc in gt_locations if loc in detected_locations]
    false_negatives = [loc for loc in gt_locations if loc not in detected_locations]
    false_positives = [loc for loc in detected_locations if loc not in gt_locations]
    
    # Overall detection (did we find ANY aneurysm when there IS one?)
    aneurysm_prob = result.get("summary", {}).get("aneurysm_probability", 0)
    predicted_positive = aneurysm_prob > 0.5
    
    return {
        "series_id": series_id,
        "ground_truth": gt_locations,
        "is_positive": is_positive,
        "aneurysm_probability": aneurysm_prob,
        "predicted_positive": predicted_positive,
        "predictions": {k: v["probability"] for k, v in all_preds.items()},
        "detected_locations": detected_locations,
        "correctly_detected": correctly_detected,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "num_gt": len(gt_locations),
        "num_detected": len(detected_locations),
        "num_correct": len(correctly_detected),
    }


def run_evaluation(series_folder: Path, output_folder: Path, max_series: int = None):
    """Run evaluation on all series in folder."""
    
    print("=" * 70)
    print("🧠 ANEURYSM DETECTION MODEL EVALUATION")
    print("=" * 70)
    print(f"📁 Series folder: {series_folder}")
    print(f"📊 Ground truth: {len(GROUND_TRUTH)} annotations in {len(POSITIVE_SERIES)} series")
    print()
    
    # Initialize detector
    detector = AneurysmDetector(device='cuda')
    
    if not detector.is_loaded:
        print("❌ Failed to load model")
        return
    
    # Get all series folders
    all_series = [d for d in series_folder.iterdir() if d.is_dir()]
    
    if max_series:
        all_series = all_series[:max_series]
    
    print(f"📂 Processing {len(all_series)} series...")
    print("-" * 70)
    
    results = []
    
    # Metrics accumulators
    tp_series = 0  # True positive: has aneurysm, detected aneurysm
    tn_series = 0  # True negative: no aneurysm, no detection
    fp_series = 0  # False positive: no aneurysm, but detected
    fn_series = 0  # False negative: has aneurysm, not detected
    
    total_gt_locations = 0
    total_correct_locations = 0
    total_detected_locations = 0
    
    for i, series_path in enumerate(all_series):
        series_id = series_path.name
        print(f"\n[{i+1}/{len(all_series)}] {series_id[:40]}...")
        
        result = evaluate_single_series(detector, series_path, series_id)
        results.append(result)
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        # Update metrics
        is_pos = result["is_positive"]
        pred_pos = result["predicted_positive"]
        
        if is_pos and pred_pos:
            tp_series += 1
        elif is_pos and not pred_pos:
            fn_series += 1
        elif not is_pos and pred_pos:
            fp_series += 1
        else:
            tn_series += 1
        
        total_gt_locations += result["num_gt"]
        total_correct_locations += result["num_correct"]
        total_detected_locations += result["num_detected"]
        
        # Print result
        prob = result["aneurysm_probability"]
        status = "✅" if (is_pos == pred_pos) else "❌"
        
        gt_str = ", ".join(result["ground_truth"]) if result["ground_truth"] else "None"
        det_str = ", ".join(result["detected_locations"]) if result["detected_locations"] else "None"
        
        print(f"   {status} GT: {gt_str[:50]}")
        print(f"      Pred ({prob:.1%}): {det_str[:50]}")
        
        if result["correctly_detected"]:
            print(f"      ✓ Correct: {', '.join(result['correctly_detected'])}")
        if result["false_negatives"]:
            print(f"      ✗ Missed: {', '.join(result['false_negatives'])}")
        if result["false_positives"]:
            print(f"      ⚠ False Pos: {', '.join(result['false_positives'])}")
    
    # Calculate final metrics
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    total_series = len(results)
    
    # Series-level metrics (presence/absence)
    accuracy = (tp_series + tn_series) / total_series if total_series > 0 else 0
    precision = tp_series / (tp_series + fp_series) if (tp_series + fp_series) > 0 else 0
    recall = tp_series / (tp_series + fn_series) if (tp_series + fn_series) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Series-Level Metrics (Aneurysm Present?)")
    print(f"   ├─ Total Series:     {total_series}")
    print(f"   ├─ True Positives:   {tp_series}")
    print(f"   ├─ True Negatives:   {tn_series}")
    print(f"   ├─ False Positives:  {fp_series}")
    print(f"   ├─ False Negatives:  {fn_series}")
    print(f"   ├─ Accuracy:         {accuracy:.1%}")
    print(f"   ├─ Precision:        {precision:.1%}")
    print(f"   ├─ Recall:           {recall:.1%}")
    print(f"   └─ F1 Score:         {f1:.3f}")
    
    # Location-level metrics
    loc_precision = total_correct_locations / total_detected_locations if total_detected_locations > 0 else 0
    loc_recall = total_correct_locations / total_gt_locations if total_gt_locations > 0 else 0
    loc_f1 = 2 * loc_precision * loc_recall / (loc_precision + loc_recall) if (loc_precision + loc_recall) > 0 else 0
    
    print(f"\n📍 Location-Level Metrics (Correct Location?)")
    print(f"   ├─ Ground Truth:     {total_gt_locations} locations")
    print(f"   ├─ Detected:         {total_detected_locations} locations")
    print(f"   ├─ Correct:          {total_correct_locations} locations")
    print(f"   ├─ Precision:        {loc_precision:.1%}")
    print(f"   ├─ Recall:           {loc_recall:.1%}")
    print(f"   └─ F1 Score:         {loc_f1:.3f}")
    
    # Save results
    output_folder.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "total_series": total_series,
        "series_metrics": {
            "true_positives": tp_series,
            "true_negatives": tn_series,
            "false_positives": fp_series,
            "false_negatives": fn_series,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "location_metrics": {
            "total_ground_truth": total_gt_locations,
            "total_detected": total_detected_locations,
            "total_correct": total_correct_locations,
            "precision": round(loc_precision, 4),
            "recall": round(loc_recall, 4),
            "f1_score": round(loc_f1, 4),
        },
        "per_series_results": results,
    }
    
    output_json = output_folder / "evaluation_results.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_json}")
    print("=" * 70)
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate aneurysm detection model')
    parser.add_argument('--series-folder', type=str, 
                        default=r"C:\Users\Rayan\Desktop\Main Project\series",
                        help='Path to folder containing DICOM series')
    parser.add_argument('--output', type=str, default='./evaluation_output',
                        help='Output folder for results')
    parser.add_argument('--max-series', type=int, default=None,
                        help='Max number of series to process (for testing)')
    
    args = parser.parse_args()
    
    run_evaluation(
        series_folder=Path(args.series_folder),
        output_folder=Path(args.output),
        max_series=args.max_series
    )


if __name__ == "__main__":
    main()
