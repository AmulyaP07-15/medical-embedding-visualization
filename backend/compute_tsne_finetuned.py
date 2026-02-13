"""
Compute t-SNE coordinates for fine-tuned embeddings
"""

import numpy as np
from sklearn.manifold import TSNE

print("Loading fine-tuned embeddings...")
embeddings = np.load('cache/embeddings_triplet-biobert.npy')
print(f"✓ Embeddings shape: {embeddings.shape}")

print("Computing t-SNE (this may take 3-5 minutes)...")
tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
coords_3d = tsne.fit_transform(embeddings)

print(f"✓ t-SNE coordinates shape: {coords_3d.shape}")

# Save
np.save('cache/tsne_triplet-biobert.npy', coords_3d)
print("✓ Saved to: cache/tsne_triplet-biobert.npy")
