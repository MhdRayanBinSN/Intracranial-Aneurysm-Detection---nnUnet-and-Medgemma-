
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # nnU-Net summary.json structure often has 'foreground_mean' or 'mean' 
    # But usually TP/FP/FN/TN are metric_per_case. 
    # For a global matrix, we sum them up up.
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    # TN is tricky in segmentation (it's huge), usually we ignore it for dice/iou
    # But if it's there, we use it.
    
    if 'metric_per_case' in data:
        for case in data['metric_per_case']:
            metrics = case['metrics'].get('1', {}) # Assuming class 1
            tp_total += metrics.get('TP', 0)
            fp_total += metrics.get('FP', 0)
            fn_total += metrics.get('FN', 0)
            # tn_total += metrics.get('TN', 0) 
            
    return tp_total, fp_total, fn_total

def plot_confusion_matrix(tp, fp, fn, tn=None):
    # If TN is missing (common in segmentation), we can leave it or estimate
    if tn is None:
        tn = 0 # Or represent as N/A
        
    # Construct Matrix
    #           Pred +   Pred -
    # Actual +   TP       FN
    # Actual -   FP       TN
    
    matrix = np.array([[tp, fn], [fp, tn]])
    
    group_names = ['True Pos','False Neg','False Pos','True Neg']
    group_counts = ["{0:0.0f}".format(value) for value in matrix.flatten()]
    
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figtext(0.5, 0.01, f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    return plt

if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Loading metrics from {sys.argv[1]}...")
        tp, fp, fn = load_metrics_from_json(sys.argv[1])
        tn = 0 # Usually not tracked or 0 for segmentation background
    else:
        print("⚠️ No summary.json provided. Showing DEMO Matrix.")
        print("Usage: python plot_confusion_matrix.py path/to/summary.json")
        # Dummy Data for Demo
        tp, fp, fn, tn = 1500, 230, 150, 50000

    print(f"Plotting: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    plt = plot_confusion_matrix(tp, fp, fn, tn)
    
    output_file = "confusion_matrix.png"
    plt.savefig(output_file)
    print(f"✅ Saved matrix to {os.path.abspath(output_file)}")
    plt.show()
