"""
Evaluate improvement: Original BioBERT vs Fine-tuned BioBERT
"""

import numpy as np
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_model(embeddings, labels, model_name):
    """Evaluate embedding quality"""
    
    # 1. Silhouette score (higher = better)
    silhouette = silhouette_score(embeddings, labels)
    
    # 2. Davies-Bouldin index (lower = better)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    # 3. Precision@5
    precisions = []
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        sims = cosine_similarity([emb], embeddings)[0]
        top5_indices = np.argsort(sims)[::-1][1:6]  # Exclude self
        matches = sum(1 for idx in top5_indices if labels[idx] == label)
        precisions.append(matches / 5)
    
    precision_at_5 = np.mean(precisions)
    
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"Silhouette Score:      {silhouette:.4f}  (higher = better)")
    print(f"Davies-Bouldin Index:  {davies_bouldin:.4f}  (lower = better)")
    print(f"Precision@5:           {precision_at_5:.4f}  (higher = better)")
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'precision_at_5': precision_at_5
    }


# Load data
print("Loading data...")
with open('cache/data_info.pkl', 'rb') as f:
    data = pickle.load(f)

labels = data['specialties']

print("\n" + "="*70)
print("COMPARISON: Original vs Fine-tuned BioBERT")
print("="*70)

# Original BioBERT
print("\nLoading original BioBERT embeddings...")
biobert_emb = np.load('cache/embeddings_biobert.npy')
original_metrics = evaluate_model(biobert_emb, labels, "Original BioBERT")

# Fine-tuned BioBERT
print("\nLoading fine-tuned BioBERT embeddings...")
triplet_emb = np.load('cache/embeddings_triplet-biobert.npy')
finetuned_metrics = evaluate_model(triplet_emb, labels, "Fine-tuned BioBERT (Triplet Loss)")

# Calculate improvement
print(f"\n{'='*70}")
print("IMPROVEMENT")
print(f"{'='*70}")

silhouette_improvement = ((finetuned_metrics['silhouette'] - original_metrics['silhouette']) 
                         / original_metrics['silhouette'] * 100)
db_improvement = ((original_metrics['davies_bouldin'] - finetuned_metrics['davies_bouldin']) 
                 / original_metrics['davies_bouldin'] * 100)
p5_improvement = ((finetuned_metrics['precision_at_5'] - original_metrics['precision_at_5']) 
                 / original_metrics['precision_at_5'] * 100)

print(f"Silhouette:     {silhouette_improvement:+.1f}%")
print(f"Davies-Bouldin: {db_improvement:+.1f}% (improvement = reduction)")
print(f"Precision@5:    {p5_improvement:+.1f}%")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
if silhouette_improvement > 0 and p5_improvement > 0:
    print("✅ Fine-tuning with triplet loss IMPROVED embedding quality!")
    print(f"   Specialty clustering is now {abs(silhouette_improvement):.0f}% better.")
    print(f"   Retrieval accuracy improved by {abs(p5_improvement):.0f}%.")
else:
    print("⚠ Mixed results - some metrics improved, others didn't.")
    print("   This can happen with small datasets or few training epochs.")
