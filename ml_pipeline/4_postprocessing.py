"""
4. Postprocessing Module
========================
Extracts detection results from probability maps.

Pipeline:
    Probability Maps → Extract Max → Apply Threshold → Format Results
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

from config import LOCATION_LABELS, LOCATION_SHORT, DETECTION_THRESHOLD


def extract_per_location_probability(probabilities: torch.Tensor) -> Dict[str, float]:
    """
    Extract maximum probability for each anatomical location.
    
    The model outputs a probability map for each location.
    We take the maximum value in each map as the "detection confidence"
    for that location.
    
    Args:
        probabilities: Tensor (num_classes, Z, Y, X)
        
    Returns:
        Dict mapping location name to max probability
    """
    results = {}
    
    for i, location in enumerate(LOCATION_LABELS):
        # Get max probability for this location
        max_prob = probabilities[i].max().item()
        results[location] = round(max_prob, 4)
    
    return results


def find_peak_coordinates(prob_map: torch.Tensor, threshold: float = 0.3) -> List[Tuple[int, int, int]]:
    """
    Find (z, y, x) coordinates of high-probability regions.
    
    Args:
        prob_map: 3D probability map for one location
        threshold: Minimum probability to consider
        
    Returns:
        List of (z, y, x) peak coordinates
    """
    # Find all voxels above threshold
    mask = prob_map > threshold
    
    if not mask.any():
        return []
    
    # Find the maximum location
    max_idx = prob_map.argmax().item()
    z, y, x = np.unravel_index(max_idx, prob_map.shape)
    
    return [(int(z), int(y), int(x))]


def classify_detections(per_location_probs: Dict[str, float],
                        threshold: float = DETECTION_THRESHOLD) -> Dict:
    """
    Classify detections into positive, negative, and uncertain.
    
    Args:
        per_location_probs: Dict of location → probability
        threshold: Detection threshold (default 50%)
        
    Returns:
        Dict with classification results
    """
    positive = []
    uncertain = []
    negative = []
    
    for location, prob in per_location_probs.items():
        if prob >= threshold:
            positive.append({
                'location': location,
                'short_name': LOCATION_SHORT.get(location, location),
                'probability': prob,
            })
        elif prob >= threshold * 0.6:  # 30-50% = uncertain
            uncertain.append({
                'location': location,
                'short_name': LOCATION_SHORT.get(location, location),
                'probability': prob,
            })
        else:
            negative.append({
                'location': location,
                'probability': prob,
            })
    
    # Sort by probability (highest first)
    positive.sort(key=lambda x: x['probability'], reverse=True)
    uncertain.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'positive': positive,
        'uncertain': uncertain,
        'negative': negative,
    }


def get_risk_level(max_probability: float) -> str:
    """
    Determine risk level based on maximum probability.
    
    Levels:
        HIGH:     ≥ 70%  - Strong evidence of aneurysm
        MODERATE: ≥ 50%  - Likely aneurysm, needs review
        LOW:      ≥ 30%  - Possible finding, low confidence
        MINIMAL:  < 30%  - No significant findings
    """
    if max_probability >= 0.70:
        return "HIGH"
    elif max_probability >= 0.50:
        return "MODERATE"
    elif max_probability >= 0.30:
        return "LOW"
    else:
        return "MINIMAL"


def postprocess(probabilities: torch.Tensor) -> Dict:
    """
    Full postprocessing pipeline.
    
    Args:
        probabilities: Model output (num_classes, Z, Y, X)
        
    Returns:
        Dict with full detection results
    """
    print("=" * 50)
    print("POSTPROCESSING")
    print("=" * 50)
    
    # Step 1: Extract per-location probabilities
    per_location = extract_per_location_probability(probabilities)
    print("Per-location probabilities:")
    for loc, prob in per_location.items():
        if prob >= 0.1:  # Only show significant ones
            print(f"  {LOCATION_SHORT.get(loc, loc)}: {prob:.1%}")
    
    # Step 2: Get overall max probability
    max_prob = max(per_location.values())
    max_location = max(per_location.items(), key=lambda x: x[1])[0]
    
    # Step 3: Classify detections
    classifications = classify_detections(per_location)
    
    # Step 4: Determine risk level
    risk_level = get_risk_level(max_prob)
    
    # Step 5: Create detection list
    detections = []
    for det in classifications['positive']:
        # Find peak coordinate
        loc_idx = LOCATION_LABELS.index(det['location'])
        peaks = find_peak_coordinates(probabilities[loc_idx])
        
        detections.append({
            'location': det['location'],
            'short_name': det['short_name'],
            'probability': det['probability'],
            'coordinates': peaks[0] if peaks else None,
        })
    
    # Compile final results
    results = {
        'aneurysm_detected': len(classifications['positive']) > 0,
        'max_probability': round(max_prob, 4),
        'max_location': max_location,
        'risk_level': risk_level,
        'num_detections': len(classifications['positive']),
        'detections': detections,
        'uncertain_findings': classifications['uncertain'],
        'per_location_probabilities': per_location,
    }
    
    print("=" * 50)
    print(f"Detection: {'POSITIVE' if results['aneurysm_detected'] else 'NEGATIVE'}")
    print(f"Risk Level: {risk_level}")
    print(f"Max Probability: {max_prob:.1%} at {LOCATION_SHORT.get(max_location, max_location)}")
    print("=" * 50)
    
    return results


def format_results_for_display(results: Dict) -> str:
    """
    Format results as human-readable string.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("ANEURYSM DETECTION RESULTS")
    lines.append("=" * 50)
    
    if results['aneurysm_detected']:
        lines.append(f"⚠️  ANEURYSM DETECTED")
        lines.append(f"Risk Level: {results['risk_level']}")
        lines.append("")
        lines.append("Detections:")
        for det in results['detections']:
            lines.append(f"  • {det['short_name']}: {det['probability']:.1%}")
    else:
        lines.append(f"✓ No aneurysm detected")
        lines.append(f"Max probability: {results['max_probability']:.1%}")
    
    if results['uncertain_findings']:
        lines.append("")
        lines.append("Uncertain findings (review recommended):")
        for unc in results['uncertain_findings']:
            lines.append(f"  ? {unc['short_name']}: {unc['probability']:.1%}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Test with dummy probabilities
    dummy_probs = torch.rand(13, 64, 64, 64) * 0.3  # Mostly low
    dummy_probs[8, 30:35, 30:35, 30:35] = 0.75  # High prob region at R-MCA
    
    results = postprocess(dummy_probs)
    print(format_results_for_display(results))
