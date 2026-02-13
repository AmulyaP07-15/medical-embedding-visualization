import pandas as pd
import numpy as np
import pickle
import os
from data_loader import load_mtsamples, filter_by_specialties
from embedding_models import load_all_models

# Define the 6 specialties we want
SELECTED_SPECIALTIES = [
    'Surgery',
    'Cardiovascular / Pulmonary',
    'Orthopedic',
    'Radiology',
    'Neurology',
    'Gastroenterology'
]

def generate_and_cache_embeddings(n_samples_per_specialty=150):
    """
    Generate embeddings for all models and cache them
    """
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    
    # Load data
    df = load_mtsamples()
    
    # Filter to selected specialties
    df_filtered = filter_by_specialties(df, SELECTED_SPECIALTIES, n_samples_per_specialty)
    
    # Save filtered data info
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    data_info = {
        'texts': df_filtered['description'].tolist(),
        'specialties': df_filtered['medical_specialty'].tolist(),
        'sample_names': df_filtered['sample_name'].tolist() if 'sample_name' in df_filtered.columns else None
    }
    
    with open(f'{cache_dir}/data_info.pkl', 'wb') as f:
        pickle.dump(data_info, f)
    
    print(f"\n✓ Cached {len(data_info['texts'])} samples")
    
    print("\n" + "=" * 60)
    print("STEP 2: Loading models")
    print("=" * 60)
    
    models = load_all_models()
    
    print("\n" + "=" * 60)
    print("STEP 3: Generating embeddings")
    print("=" * 60)
    
    texts = data_info['texts']
    
    for model_name, model in models.items():
        print(f"\n>>> Generating embeddings with {model.name}")
        embeddings = model.encode(texts)
        
        # Save embeddings
        cache_path = f'{cache_dir}/embeddings_{model_name}.npy'
        np.save(cache_path, embeddings)
        
        print(f"✓ Saved {model_name} embeddings: {embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("✓ ALL EMBEDDINGS GENERATED AND CACHED")
    print("=" * 60)
    print(f"\nCache location: {os.path.abspath(cache_dir)}")
    print(f"Total samples: {len(texts)}")
    print(f"Specialties: {len(SELECTED_SPECIALTIES)}")

if __name__ == "__main__":
    generate_and_cache_embeddings(n_samples_per_specialty=150)
