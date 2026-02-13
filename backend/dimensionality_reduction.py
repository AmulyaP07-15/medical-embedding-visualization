import numpy as np
from sklearn.manifold import TSNE
import umap
import pickle

def reduce_to_3d_tsne(embeddings, perplexity=30, random_state=42):
    """
    Reduce embeddings to 3D using t-SNE
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        perplexity: t-SNE perplexity parameter
        random_state: random seed for reproducibility
    """
    print(f"Running t-SNE on {embeddings.shape[0]} samples...")
    
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    
    reduced = tsne.fit_transform(embeddings)
    print(f"✓ t-SNE complete: {reduced.shape}")
    
    return reduced

def reduce_to_3d_umap(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Reduce embeddings to 3D using UMAP
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: random seed for reproducibility
    """
    print(f"Running UMAP on {embeddings.shape[0]} samples...")
    
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    
    reduced = reducer.fit_transform(embeddings)
    print(f"✓ UMAP complete: {reduced.shape}")
    
    return reduced

def precompute_tsne_for_all_models():
    """
    Pre-compute t-SNE for all cached embedding models
    """
    import os
    
    cache_dir = './cache'
    embedding_files = [f for f in os.listdir(cache_dir) if f.startswith('embeddings_') and f.endswith('.npy')]
    
    print("=" * 60)
    print("PRE-COMPUTING t-SNE FOR ALL MODELS")
    print("=" * 60)
    
    for emb_file in embedding_files:
        model_name = emb_file.replace('embeddings_', '').replace('.npy', '')
        print(f"\n>>> Processing {model_name}")
        
        # Load embeddings
        embeddings = np.load(f'{cache_dir}/{emb_file}')
        print(f"Loaded embeddings: {embeddings.shape}")
        
        # Compute t-SNE
        tsne_coords = reduce_to_3d_tsne(embeddings)
        
        # Save t-SNE coordinates
        output_path = f'{cache_dir}/tsne_{model_name}.npy'
        np.save(output_path, tsne_coords)
        print(f"✓ Saved t-SNE to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✓ ALL t-SNE COMPUTATIONS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    precompute_tsne_for_all_models()
